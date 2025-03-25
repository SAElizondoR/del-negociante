import os
import time
import ccxt
import pandas as pd
import numpy as np
import ta
import xgboost as xgb
from dotenv import load_dotenv
from datetime import datetime, timedelta
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm
import sqlite3
from sklearn.model_selection import RandomizedSearchCV

# Configuración
load_dotenv()
altcoin_symbol = 'SOL/USDT'
benchmark_symbols = []
timeframe = '4h'
initial_capital = 1000000.0
commission = 0.00075
DATA_CACHE = 'market_data_4h.db'

class CrossAssetTradingSystem:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.scaler = RobustScaler()
        self.portfolio_history = []
        self.signal_history = []
        # Nuevo: Historial de datos de riesgo
        self.risk_history = []

        self.model = xgb.XGBClassifier(
            n_estimators=1000,  # Aumentar y usar early stopping
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss',
            early_stopping_rounds=50,
            subsample=0.7,
            min_child_weight=10,
            max_depth=5,
            learning_rate=0.05,
            gamma=0.2,
            colsample_bytree=0.8
        )
        self.risk_params = {
            'max_exposure': 0.15,  # Máximo 15% del capital por trade
            'profit_target_ratio': 2.5,  # Relación riesgo/recompensa
            'max_trade_duration': 24  # Horas máximo de trade abierto
        }
        self.trade_stats = []

        self.hold_period = 6  # En horas
        self.data_conn = sqlite3.connect(DATA_CACHE)
        self._init_database()
        self.reset()
    
    def _init_database(self):
        cursor = self.data_conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ohlcv (
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                PRIMARY KEY (symbol, timestamp)
            );
        ''')
        self.data_conn.commit()

    def reset(self):
        self.portfolio = {
            'cash': initial_capital,
            'altcoin': 0.0,
            'value': initial_capital
        }
        self.trades = []
        self.open_trades = []
        self.feature_columns = []
        self.raw_columns = []
        self.portfolio_history = []
        # Reiniciar histórico de señales
        self.signal_history = []
        # Nuevo: Reiniciar el historial de riesgo
        self.risk_history = []
    
    def record_portfolio_value(self, df, index, timestamp):
        """Registra de forma sistemática el valor del portafolio en el tiempo."""
        self.portfolio_history.append({
            'timestamp': timestamp,
            'cash': self.portfolio['cash'],
            'altcoin': self.portfolio['altcoin'],
            'value': self.portfolio['value'],
            'exposure': sum(t['size']*df.iloc[index]['SOL_close'] for t in self.open_trades)
        })
    
    def fetch_and_cache_data(self, days=365*3):  # 3 años de datos
        exchange = ccxt.binance({'enableRateLimit': True})
        cursor = self.data_conn.cursor()

        for symbol in [altcoin_symbol] + benchmark_symbols:
            since = exchange.parse8601((datetime.now() - timedelta(days=days)).isoformat())
            all_ohlcv = []

            while True:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since)
                if not ohlcv:
                    break
                all_ohlcv.extend(ohlcv)
                since = ohlcv[-1][0] + 1  # Siguiente intervalo
                print(f"Descargados {len(ohlcv)} registros para {symbol}")
                time.sleep(exchange.rateLimit / 1000)  # Respetar rate limit
            
            # Almacenar en base de datos
            data = [(
                symbol,
                row[0],
                row[1],
                row[2],
                row[3],
                row[4],
                row[5]
            ) for row in all_ohlcv]
            
            cursor.executemany('''
                INSERT OR IGNORE INTO ohlcv (symbol, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', data)
            self.data_conn.commit()
    
    def load_cached_data(self):
        query = f'''
            SELECT 
                timestamp,
                symbol,
                open,
                high,
                low,
                close,
                volume
            FROM ohlcv
            WHERE symbol IN ({','.join(['?'] * (len(benchmark_symbols) + 1))})
            ORDER BY timestamp
        '''
        df = pd.read_sql(query, self.data_conn, 
                        params=[altcoin_symbol] + benchmark_symbols)
        
        # Reestructurar datos multi-símbolo
        multi_df = df.pivot(index='timestamp', columns='symbol', 
                           values=['open', 'high', 'low', 'close', 'volume'])
        multi_df.columns = [f"{col[1].replace('/','_')}_{col[0]}" 
                          for col in multi_df.columns]
        return self.calculate_cross_features(multi_df.ffill().dropna())
    
    # Mejorado: Target adaptativo basado en volatilidad
    def calculate_target(self, prices, atr_series):
        targets = []
        lookahead = 6
        for i in range(len(prices) - lookahead):
            atr = atr_series.iloc[i]
            upper = prices.iloc[i] + 0.8 * atr  # 0.8xATR como objetivo
            lower = prices.iloc[i] - 0.5 * atr  # 0.5xATR como stop
            future = prices.iloc[i:i+lookahead]
            
            if (future > upper).any():
                targets.append(2)  # Compra
            elif (future < lower).any():
                targets.append(0)  # Venta
            else:
                targets.append(1)  # Neutral
        return pd.Series(targets, index=prices.index[:len(targets)])

    def calculate_cross_features(self, df):
        df.columns = df.columns.str.replace('USDT_', '')
        
        # ===== FEATURES PARA SOL =====
        sol_df = df.filter(like='SOL').copy()
        sol_df.columns = sol_df.columns.str.replace('SOL_', '')
        
        # Indicadores base
        df['SOL_returns'] = sol_df['close'].pct_change()
        df['SOL_atr_raw'] = ta.volatility.average_true_range(sol_df['high'], sol_df['low'], sol_df['close'], 14)
        df['SOL_rsi_14h'] = ta.momentum.rsi(sol_df['close'], 14)  # Corregido el nombre
        df['SOL_obv'] = ta.volume.on_balance_volume(sol_df['close'], sol_df['volume'])
        
        # Features temporales requeridas por el signal generator
        df['SOL_volume_ma_24h'] = sol_df['volume'].rolling(24).mean()
        df['SOL_high_24h'] = sol_df['high'].rolling(24).max()
        df['SOL_low_24h'] = sol_df['low'].rolling(24).min()
        df['SOL_volatility_1h'] = sol_df['close'].pct_change().rolling(4).std()  # Para el volatility exit
        
        # ===== FEATURES PARA LOS BENCHMARKS =====
        benchmark_symbols = []  # Asumidos de contexto anterior
        
        for symbol in benchmark_symbols:
            base = symbol.split('/')[0]
            close_col = f'{base}_close'
            
            # Features base
            df[f'{base}_ret_1h'] = df[close_col].pct_change()
            df[f'{base}_ret_24h'] = df[close_col].pct_change(24)
            df[f'{base}_ma_50_raw'] = df[close_col].rolling(50).mean()
            df[f'{base}_ma_200_raw'] = df[close_col].rolling(200).mean()
            df[f'{base}_corr_24h_raw'] = df['SOL_close'].rolling(24).corr(df[close_col])
            
            # Nueva feature de volatilidad para ETH requerida en market regime
            df[f'{base}_volatility_24h'] = df[f'{base}_ret_1h'].rolling(24).std()
        
        # ===== TARGET =====
        df['target'] = self.calculate_target(df['SOL_close'], df['SOL_atr_raw'])
        
        # ===== GESTIÓN DE COLUMNAS =====
        self.raw_columns = [
            'SOL_close', 'SOL_high', 'SOL_low', 'SOL_volume',
            'SOL_volume_ma_24h', 'SOL_high_24h', 'SOL_low_24h',
            'SOL_rsi_14h', 'SOL_atr_raw', 'SOL_volatility_1h'
        ]
        
        # Columnas para el modelo (excluyendo raw columns y target)
        self.feature_columns = [col for col in df.columns if col not in self.raw_columns + ['target', 'model_proba']]
        
        # Escalado solo de features del modelo
        df[self.feature_columns] = self.scaler.fit_transform(df[self.feature_columns])
        
        # ===== VALIDACIÓN =====
        if self.verbose:
            print("\n=== Feature Engineering Report ===")
            print("Nuevas Features Críticas:")
            print(f"- SOL High 24h: {df['SOL_high_24h'].iloc[-1]:.4f}")
            print(f"- SOL Volume MA 24h: {df['SOL_volume_ma_24h'].iloc[-1]:.2f}")
            # print(f"- ETH Volatility 24h: {df['ETH_volatility_24h'].iloc[-1]:.4f}")
            # print(f"- BTC/ETH Correlation Spread: {df['BTC_corr_24h_raw'].iloc[-1] - df['ETH_corr_24h_raw'].iloc[-1]:.2f}")
            print("\nDistribución de Features Escaladas:")
            print(df[self.feature_columns].iloc[-1].describe())

        return df.dropna()

    def train_model(self, train_data):
        X = train_data[self.feature_columns]
        y = train_data['target']

        param_dist = {
            'max_depth': [5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'min_child_weight': [1, 5, 10],
            'gamma': [0, 0.1, 0.2]
        }
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Validación temporal estricta
        train_size = int(len(X) * 0.8)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        # Aplicar scaling correctamente
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        search = RandomizedSearchCV(self.model, param_dist, n_iter=50, cv=tscv, scoring='neg_log_loss')
        # search.fit(
        #     X_train_scaled, y_train,
        #     eval_set=[(X_val_scaled, y_val)]
        # )
        # self.model = search.best_estimator_
        
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val)],
            verbose=self.verbose
        )

    def generate_signal(self, current_data, df, current_index):
        X = current_data[self.feature_columns].values.reshape(1, -1)
        df.loc[current_index, 'model_proba'] = proba = self.model.predict_proba(X)[0][1]
        
        # 1. Condiciones de Mercado Global
        # btc_trend = current_data['BTC_ma_50_raw'] > current_data['BTC_ma_200_raw']
        
        # 2. Dinámica de Threshold Adaptativo
        volatility_factor = current_data['SOL_atr_raw'] / current_data['SOL_close']
        liquidity_factor = current_data['SOL_volume'] / current_data['SOL_volume_ma_24h']
        dynamic_threshold = 0.65 + (volatility_factor * 15) - (liquidity_factor * 0.2)
        dynamic_threshold = np.clip(dynamic_threshold, 0.55, 0.85)
        
        # 3. Confirmadores Técnicos
        rsi = current_data['SOL_rsi_14h']
        volume_ratio = current_data['SOL_volume'] / current_data['SOL_volume_ma_24h']
        price_position = (current_data['SOL_close'] - current_data['SOL_low_24h']) / \
                        (current_data['SOL_high_24h'] - current_data['SOL_low_24h'])
        
        # 4. Condiciones Combinadas
        long_cond = all([
            proba > dynamic_threshold,
            rsi < 65,
            volume_ratio > 1.2,
            price_position > 0.5,
            volatility_factor < 0.035,
        ])
        
        short_cond = all([
            proba < (1 - dynamic_threshold),
            rsi > 35,
            volume_ratio > 1.1,
            price_position < 0.4,
            volatility_factor < 0.04,
        ])
        
        # 5. Filtro de Confirmación Temporal
        # if long_cond or short_cond:
        #     # Verificar consistencia en las últimas 3 velas
        #     lookback = 3
        #     historical_proba = df.iloc[current_index-lookback:current_index]['model_proba']
        #     confirm_trend = np.all(np.diff(historical_proba) > 0) if long_cond else \
        #                   np.all(np.diff(historical_proba) < 0)
            
        #     if not confirm_trend:
        #         return 0, proba
        
        # # 6. Gestión de Riesgo Dinámica
        # if self.calculate_portfolio_risk() > self.risk_params['max_exposure']:
        #     return 0, proba
        
        # 7. Generación de Señal Final
        signal = 1 if long_cond else -1 if short_cond else 0

        # Nuevo: Registrar en el histórico de señales los detalles y cumplimiento de reglas
        self.signal_history.append({
            'timestamp': current_data.index,
            'signal': signal,
            'model_proba': proba,
            'dynamic_threshold': dynamic_threshold,
            # 'btc_trend': btc_trend,
            'rsi': rsi,
            'volume_ratio': volume_ratio,
            'price_position': price_position,
            'volatility_factor': volatility_factor,
            # Cumplimiento de reglas para señal LONG
            'long_rule1': proba > dynamic_threshold,
            'long_rule2': rsi < 65,
            'long_rule3': volume_ratio > 1.2,
            'long_rule4': price_position > 0.5,
            'long_rule5': volatility_factor < 0.035,
            # Cumplimiento de reglas para señal SHORT
            'short_rule1': proba < (1 - dynamic_threshold),
            'short_rule2': rsi > 35,
            'short_rule3': volume_ratio > 1.1,
            'short_rule4': price_position < 0.4,
            'short_rule5': volatility_factor < 0.04
        })
        
        if self.verbose:
            self._log_signal_details(current_data, proba, dynamic_threshold, 
                                    long_cond, short_cond)
        
        return signal, proba
    
    def calculate_portfolio_risk(self, current_price):
        """
        Calcula el porcentaje del capital que está en riesgo.
        Se multiplica el tamaño de cada trade abierto por el precio actual para obtener su valor,
        se suma el valor de todas las posiciones abiertas y se divide entre el capital total
        (valor actual de la cartera).
        """
        total_capital = self.portfolio['value']
        exposure_value = sum(trade['size'] * current_price for trade in self.open_trades)
        risk_percentage = (exposure_value / total_capital) * 100  # Expresado en porcentaje
        return risk_percentage
    
    def _log_signal_details(self, data, proba, threshold, long_cond, short_cond):
        regime="a"
        print("\n=== Signal Generation Details ===")
        print(f"Market Regime: {regime.upper()}")
        print(f"Model Probability: {proba:.2%} | Dynamic Threshold: {threshold:.2%}")
        # print(f"BTC Trend: {'Bull' if data['BTC_ma_50_raw'] > data['BTC_ma_200_raw'] else 'Bear'}")
        # print(f"ETH Correlation: {data['ETH_corr_24h_raw']:.2f}")
        print(f"Volatility (SOL): {data['SOL_atr_raw']/data['SOL_close']:.2%}")
        print(f"Volume Ratio: {data['SOL_volume']/data['SOL_volume_ma_24h']:.2f}")
        print(f"RSI Position: {data['SOL_rsi_14h']:.1f}")
        print(f"Conditions Met - LONG: {long_cond} | SHORT: {short_cond}")

    def _log_trade_close(self, trade):
        print("\n=== Position Closed ===")
        print(f"Direction: {'LONG' if trade['signal'] == 1 else 'SHORT'}")
        print(f"Duration: {trade['duration']:.1f}h | Return: {trade['return']:.2f}%")
        print(f"Exit Reasons: {trade['exit_reasons']}")

    def execute_trade(self, df, current_index):
        if self.verbose and (current_index % 100 == 0):
            print("\n=== Portfolio Status ===")
            print(f"Current Value: ${self.portfolio['value']:,.2f}")
            print(f"Open Trades: {len(self.open_trades)}")
            print(f"Cash Available: ${self.portfolio['cash']:,.2f}")
        
        current_data = df.iloc[current_index]
        signal, proba = self.generate_signal(current_data, df, current_index)

        self.evaluate_open_trades(df, current_index, proba)
        
        if signal == 0 or len(self.open_trades) >= 3:
            self.record_portfolio_value(df, current_index, df.index[current_index])
            return
        
        current_price = current_data['SOL_close']
        position_size = self.calculate_position_size(current_price)
        
        if position_size > 0:
            self.open_position(signal, current_price, position_size, df.index[current_index], current_index)
        
        self.record_portfolio_value(df, current_index, df.index[current_index])

    def evaluate_open_trades(self, df, current_index, proba):
        current_price = df.iloc[current_index]['SOL_close']
        current_time = df.index[current_index]
        current_proba = proba

        for trade in self.open_trades.copy():
            duration = (int(current_time) - int(trade['open_time'])) / 1000
            atr = df.iloc[current_index]['SOL_atr_raw']
            entry_price = trade['entry_price']
            
            # 2. Condiciones de Salida
            exit_conditions = {
                'stop_loss': (
                    (trade['signal'] == 1 and current_price <= trade['stop_loss']) or
                    (trade['signal'] == -1 and current_price >= trade['stop_loss'])
                ),
                'profit_target': (
                    (trade['signal'] == 1 and current_price >= entry_price + self.risk_params['profit_target_ratio'] * (entry_price - trade['stop_loss'])) or
                    (trade['signal'] == -1 and current_price <= entry_price - self.risk_params['profit_target_ratio'] * (trade['stop_loss'] - entry_price))
                ),
                'time_exit': duration > self.risk_params['max_trade_duration'],
                'signal_reversal': (
                    (trade['signal'] == 1 and current_proba < 0.35) or
                    (trade['signal'] == -1 and current_proba > 0.65)
                )
            }
            
            # 3. Actualización Dinámica del Stop Loss
            if trade['signal'] == 1:
                new_stop = max(
                    trade['stop_loss'],
                    current_price - 1.8 * atr,
                    entry_price * 0.995  # Protección de capital mínimo
                )
                if current_price > entry_price:
                    new_stop += (current_price - entry_price) * 0.3  # Trailing progresivo
                trade['stop_loss'] = new_stop
            else:
                new_stop = min(
                    trade['stop_loss'],
                    current_price + 1.8 * atr,
                    entry_price * 1.005
                )
                if current_price < entry_price:
                    new_stop -= (entry_price - current_price) * 0.3
                trade['stop_loss'] = new_stop
            
            # 4. Evaluación de Condiciones de Salida
            if any(exit_conditions.values()):
                self.close_position(trade, current_price, current_time)
                continue
            
            # 5. Comprobación de Correlaciones
            if abs(df.iloc[current_index]['ETH_corr_24h_raw']) < 0.25:
                self.close_position(trade, current_price, current_time)

    def calculate_position_size(self, current_price):
        risk = 0.12
        volatility = self.current_volatility()
        volatility_factor = max(0.2, 1 - (volatility / 0.05))  # Límite mínimo de 20%
        
        size = (self.portfolio['cash'] * risk * volatility_factor) / current_price
        max_size = (self.portfolio['cash'] / current_price) * 0.075
        
        return min(size, max_size)

    def current_volatility(self):
        if len(self.trades) < 5:
            return 0.02
        prices = [t['entry_price'] for t in self.trades[-5:]]
        returns = np.diff(prices) / prices[:-1]
        return np.std(returns)

    def open_position(self, signal, price, size, timestamp, open_index):
        fee = size * price * commission
        self.portfolio['cash'] -= (size * price) + fee
        self.portfolio['altcoin'] += size * signal
        trade = {
            'open_time': timestamp,
            'signal': signal,
            'size': size,
            'entry_price': price,
            'open_index': open_index,
            'stop_loss': 0.01
        }
        self.open_trades.append(trade)
        # Registrar datos de riesgo en el instante de apertura
        risk_info = {
            'timestamp': timestamp,
            'entry_price': price,
            'stop_loss': trade['stop_loss'],
            'risk_reward_ratio': self.risk_params['profit_target_ratio'],
            'exposure_pct': self.calculate_portfolio_risk(price)
        }
        self.risk_history.append(risk_info)

    def close_position(self, trade, exit_price, exit_time):
        fee = trade['size'] * exit_price * commission
        profit = (exit_price - trade['entry_price']) * trade['size']
        
        self.portfolio['cash'] += (trade['size'] * exit_price) - fee
        self.portfolio['altcoin'] -= trade['size'] * trade['signal']
        self.portfolio['value'] = self.portfolio['cash'] + self.portfolio['altcoin'] * exit_price
        
        trade['exit_price'] = exit_price
        trade['exit_time'] = exit_time
        trade['profit'] = profit

        self.trades.append(trade)
        self.open_trades.remove(trade)

    def analyze_performance(self, df):
        if self.open_trades:
            last_price = df.iloc[-1]['SOL_close']
            for trade in self.open_trades.copy():
                self.close_position(trade, last_price, df.index[-1])
        
        total_return = self.portfolio['value'] - initial_capital
        df.index = pd.to_datetime(df.index.astype(int) / 1000, unit='s')
        days = (df.index[-1] - df.index[0]).days
        print(df.index[-1])
        print(df.index[0])
        
        print("\n=== Resultados Mejorados con Análisis Cruzado ===")
        print(f"Capital Final: ${self.portfolio['value']:,.2f}")
        print(f"Retorno Total: {total_return/initial_capital:.2%}")
        if(days):
            annualized_return = (1 + total_return/initial_capital)**(365/days) - 1
            print(f"Retorno Anualizado: {annualized_return:.2%}")
        print(f"Trades Realizados: {len(self.trades)}")
        
        if self.trades:
            profits = pd.Series([t['profit'] for t in self.trades])
            wins = profits[profits > 0]
            losses = profits[profits < 0]
            
            print("\nEstadísticas Clave:")
            win_rate = len(wins)/len(profits)
            print(f"Win Rate: {win_rate:.2%}")
            profit_factor = wins.sum()/abs(losses.sum())
            print(f"Profit Factor: {profit_factor:.2f}")
            max_drawdown = abs(losses.mean()/wins.mean())
            print(f"Ratio Riesgo/Recompensa: {max_drawdown:.2f}")
            print(f"Mayor Ganancia: ${wins.max():,.2f}")
            print(f"Mayor Pérdida: ${losses.min():,.2f}")
        
        # Tabla 6: Métricas de Performance
        performance_metrics = {
            'Annualized Return': annualized_return,
            'Win Rate': win_rate,
            'Profit Factor': profit_factor,
            'Ratio Riesgo/Recompensa': max_drawdown
        }
        pd.DataFrame([performance_metrics]).to_csv('performance_metrics.csv', index=False)
    
    # Nuevo: Exportar histórico de señales para Power BI
    def export_signals_history(self):
        signals_df = pd.DataFrame(self.signal_history)
        signals_df.to_csv('signals_history.csv', index=False)
        print("Histórico de señales exportado a 'signals_history.csv'.")
    
    # Opcional: Exportar matriz de cumplimiento de reglas (agregada o por señal)
    def export_rules_compliance_matrix(self):
        signals_df = pd.DataFrame(self.signal_history)
        # Aquí se puede realizar algún procesamiento adicional para obtener una matriz
        # por ejemplo, calcular el porcentaje de cumplimiento por regla
        rules = ['long_rule1', 'long_rule2', 'long_rule3', 'long_rule4', 'long_rule5',
                 'short_rule1', 'short_rule2', 'short_rule3', 'short_rule4', 'short_rule5']
        compliance = {rule: signals_df[rule].mean() for rule in rules}
        compliance_df = pd.DataFrame(list(compliance.items()), columns=['Rule', 'Compliance'])
        compliance_df.to_csv('rules_compliance_matrix.csv', index=False)
        print("Matriz de cumplimiento de reglas exportada a 'rules_compliance_matrix.csv'.")

    
    def export_performance_metrics(self):
        """
        Esta función calcula y exporta las métricas clave de rentabilidad diaria y semanal.
        Se utiliza el registro histórico del valor del portafolio para obtener una visión
        integral y transparente de la evolución en el tiempo, en aras de la planificación
        colectiva y la rendición de cuentas.
        """
        # Convertir el historial a DataFrame y formatear las marcas temporales
        hist_df = pd.DataFrame(self.portfolio_history)
        hist_df['timestamp'] = pd.to_datetime(hist_df['timestamp'], unit='ms')
        hist_df = hist_df.sort_values('timestamp').set_index('timestamp')

        # Métricas Diarias
        daily = hist_df['value'].resample('D').agg(['first', 'last', 'max', 'min'])
        daily['daily_return'] = daily['last'] / daily['first'] - 1

        # Métricas Semanales
        weekly = hist_df['value'].resample('W').agg(['first', 'last', 'max', 'min'])
        weekly['weekly_return'] = weekly['last'] / weekly['first'] - 1

        # Exportar a CSV para su posterior análisis en Power BI
        daily.to_csv('daily_performance_metrics.csv')
        weekly.to_csv('weekly_performance_metrics.csv')
        print("Métricas de rentabilidad exportadas exitosamente.")
    
    def compute_correlation_matrix(self, df):
        # Seleccionamos las columnas de precios de cierre de SOL, BTC y ETH.
        cols = ['SOL_close']
        correlation_matrix = df[cols].corr()
        print("Matriz de correlaciones entre SOL, BTC y ETH:")
        print(correlation_matrix)
        correlation_matrix.to_csv('correlation_matrix.csv')
        return correlation_matrix

    def simulate_worst_case_drawdown(self, extension_periods=10):
        """
        Simula un escenario hipotético de stress test extendiendo la peor racha
        de pérdidas registrada en el portafolio.
        
        Parameters:
            extension_periods (int): Número de períodos adicionales a simular.
            
        Returns:
            simulation_df (DataFrame): Serie temporal con los valores hipotéticos.
        """
        # Convertir el historial del portafolio a DataFrame
        df_hist = pd.DataFrame(self.portfolio_history)
        df_hist['timestamp'] = pd.to_datetime(df_hist['timestamp'], unit='ms')
        df_hist = df_hist.sort_values('timestamp')
        
        # Calcular el máximo acumulado y el drawdown
        df_hist['running_max'] = df_hist['value'].cummax()
        df_hist['drawdown'] = (df_hist['value'] - df_hist['running_max']) / df_hist['running_max']
        
        # Se obtiene el peor drawdown (valor mínimo, negativo)
        worst_drawdown = df_hist['drawdown'].min()
        print("Worst-case drawdown observado: {:.2%}".format(worst_drawdown))
        
        # Simular extensión del escenario de pérdidas
        last_value = df_hist['value'].iloc[-1]
        simulation = []
        current_value = last_value
        
        # Para cada período de extensión se aplica el mismo % de pérdida
        for i in range(1, extension_periods + 1):
            current_value = current_value * (1 + worst_drawdown)
            simulation.append({'period': i, 'hypothetical_value': current_value})
        
        simulation_df = pd.DataFrame(simulation)
        simulation_df.to_csv('worst_case_drawdown_simulation.csv', index=False)
        print("Simulación de worst-case drawdown exportada a 'worst_case_drawdown_simulation.csv'")
        
        return simulation_df


def main():
    system = CrossAssetTradingSystem(verbose=False)
    if not os.path.exists(DATA_CACHE):
        system.fetch_and_cache_data(days=3*365)
    
    # Cargar datos y preparar características
    df = system.load_cached_data()
    
    split_idx = int(len(df) * 0.9)
    system.train_model(df.iloc[:split_idx])
    
    test_data = df.iloc[split_idx:]
    for i in tqdm(range(len(test_data)), desc="Simulando Estrategia Cruzada"):
        current_window = pd.concat([df.iloc[:split_idx], test_data.iloc[:i+1]])
        system.execute_trade(current_window, split_idx + i)
    
    system.analyze_performance(test_data)

    corr_matrix = system.compute_correlation_matrix(df)

    # Tabla 1: Historial de Trades
    trades_df = pd.DataFrame(system.trades)
    print(trades_df.head())
    trades_df['timestamp'] = pd.to_datetime(trades_df['exit_time'], unit='ms')
    trades_df.to_csv('trades_history.csv', index=False)

    # Tabla 2: Evolución del Portfolio
    pd.DataFrame(system.portfolio_history).to_csv('portfolio_history.csv', index=False)

    # Exportar las métricas de rendimiento diarias y semanales
    system.export_performance_metrics()

    # Exportar el histórico de señales y la matriz de cumplimiento de reglas
    system.export_signals_history()
    system.export_rules_compliance_matrix()

    # Tabla 3: Datos de Mercado
    market_data = df[['SOL_close', 'SOL_open', 'SOL_high', 'SOL_low', 'SOL_close', 'SOL_volume', 'SOL_rsi_14h', 'SOL_atr_raw']].reset_index()
    market_data.to_csv('market_data.csv', index=False)

    df['model_proba'] = system.model.predict_proba(df[system.feature_columns].values)[:, 1]  # Tomamos la probabilidad para la clase positiva (índice 1)

    # Tabla 4: Historial de Señales
    signals = df[['model_proba', 'SOL_close']].copy()
    signals['signal'] = df.apply(lambda x: system.generate_signal(x, df, x.name)[0], axis=1)
    signals.to_csv('signals_history.csv')

    # Tabla 5: Datos de Riesgo
    pd.DataFrame(system.risk_history).to_csv('risk_metrics.csv', index=False)

    system.simulate_worst_case_drawdown(extension_periods=4)

if __name__ == '__main__':
    main()
