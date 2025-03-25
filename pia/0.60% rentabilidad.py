import os
import ccxt
import pandas as pd
import numpy as np
import ta
from dotenv import load_dotenv
from sklearn.model_selection import TimeSeriesSplit

# Configuración mejorada
load_dotenv()
symbol = 'BTC/USDT'
timeframe = '1h'
capital_inicial = 1000000.0
commission = 0.00075
risk_per_trade = 0.3
minimum_trade_value = 5  # Valor mínimo ajustado según el exchange
max_trade_duration = 24  # Horas máximas por operación

exchange = ccxt.binance({'apiKey': os.getenv('API_KEY'), 'secret': os.getenv('API_SECRET')})

class TradingEngine:
    def __init__(self):
        self.equity = [capital_inicial]
        self.trades = []
        self.current_position = None
        self.entry_time = None
        self.entry_price = None
        self.position_size = None
        self.btc_balance = 0.0  # Nuevo: Cantidad de BTC en posesión
        self.usd_balance = capital_inicial  # Separar balance en USD y BTC
    
    def get_total_equity(self, current_price):
        """Calcula el equity total (USD + valor BTC)"""
        return self.usd_balance + (self.btc_balance * current_price)

    def get_volatility_adjusted_params(self, df):
        """Ajusta parámetros usando 1.5 veces el ATR para el trailing stop"""
        atr = df['atr'].iloc[-1]
        trailing_stop = atr / df['close'].iloc[-1] * 1.5  # 1.5x ATR
        close_std = df['close'].pct_change().std() * np.sqrt(365*24)
        return {
            'trailing_stop': max(0.015, min(trailing_stop, 0.05)),  # Entre 1.5% y 5%
            'adx_threshold': 20 if close_std < 0.4 else 15,
            'rsi_buy': max(30, 40 - (close_std * 50)),
            'rsi_sell': min(70, 60 + (close_std * 50)),
        }

    def calculate_features(self, df):
        """Calcula indicadores técnicos sin data leakage"""
        df['ema21'] = df['close'].transform(lambda x: x.ewm(span=21, min_periods=21).mean())
        df['ema50'] = df['close'].transform(lambda x: x.ewm(span=50, min_periods=50).mean())
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], 14).rsi()
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], 14).average_true_range()
        df['macd'] = ta.trend.MACD(df['close']).macd_diff()
        print("DEBUG: Features calculated. Última fila:")
        print(df.iloc[-1])
        return df.dropna()

    def generate_signal(self, df):
        """Genera señales contextuales para apertura y cierre de posición"""
        last = df.iloc[-1]
        # Si no hay posición abierta, buscar una señal de compra
        if self.current_position is None:
            # Abrir si EMA21 está cerca o por encima de EMA50 (por ejemplo, 99% de EMA50)
            if last['ema21'] >= last['ema50'] * 0.99:
                signal = 'buy'
            else:
                signal = None
        else:
            # Si ya hay posición, cerrar si EMA21 cae por debajo de EMA50 (por ejemplo, 99.5% de EMA50)
            if last['ema21'] < last['ema50'] * 0.995:
                signal = 'sell'
            else:
                signal = None
        print(f"DEBUG: Señal generada: {signal}")
        return signal

    def execute_trade(self, df):
        current_price = df['close'].iloc[-1]
        print(f"Equity Total: ${self.get_total_equity(current_price):.2f} (USD: ${self.usd_balance:.2f} | BTC: {self.btc_balance:.6f} ≈ ${self.btc_balance * current_price:.2f})")

        params = self.get_volatility_adjusted_params(df)
        
        # Cálculo de equity TOTAL para gestión de riesgo
        total_equity = self.get_total_equity(current_price)
        max_risk_amount = total_equity * risk_per_trade

        # Cálculo de posición considerando comisiones
        position_size = max_risk_amount / (current_price * (1 + commission * 2))  # Comisión entrada + salida
        
        # Calcula el tamaño máximo considerando comisión de entrada y salida
        total_commission_rate = commission * 2  # Comisión entrada + salida
        
        # Asegurar el mínimo valor de operación
        min_position_size = minimum_trade_value / (current_price * (1 + total_commission_rate))
        position_size = max(position_size, min_position_size)
        
        # Verificar que no exceda el capital disponible
        position_value = position_size * current_price * (1 + commission)
        available_equity = self.usd_balance
        if position_value > available_equity:
            position_size = available_equity / (current_price * (1 + commission))
        
        self.position_size = position_size
        print(f"DEBUG: Equity: {available_equity:.2f}, Position Size: {self.position_size:.6f}, Value: {self.position_size * current_price:.2f}")
        
        # Comentario: Desactivar el cierre forzoso por tiempo para pruebas
        # time_elapsed = (df['timestamp'].iloc[-1] - self.entry_time).total_seconds()/3600 if self.entry_time else 0
        # if self.current_position and time_elapsed > max_trade_duration:
        #     print(f"DEBUG: Cierre forzoso por tiempo. Duración: {time_elapsed:.2f} horas")
        #     self.close_position(df, current_price, 'time_exit')
        
        # Si hay una posición abierta, evaluar condiciones de cierre (trailing stop / take profit)
        if self.current_position:
            profit_pct = (current_price / self.entry_price - 1) if self.current_position == 'long' else (self.entry_price / current_price - 1)
            print(f"DEBUG: Profit % de la posición abierta: {profit_pct:.4f}")
            if profit_pct <= -params['trailing_stop'] or profit_pct >= params['trailing_stop'] * 2:
                print(f"DEBUG: Condición de cierre cumplida: profit_pct={profit_pct:.4f}, Threshold: {-params['trailing_stop']} o {params['trailing_stop']*2}")
                self.close_position(df, current_price, 'stop/take')
        
                # Apertura de nueva posición si no hay posición abierta y se genera señal
        signal = self.generate_signal(df)
        if not self.current_position and signal:
            required_value = self.position_size * current_price
            if required_value >= minimum_trade_value:
                fee = self.position_size * current_price * commission
                new_equity = available_equity - (self.position_size * current_price + fee)
                print(f"DEBUG: Abriendo posición. Señal: {signal}, Fee: {fee:.4f}, Nuevo Equity: {new_equity:.2f}")
                self.equity.append(new_equity)
                self.current_position = signal
                self.entry_time = df['timestamp'].iloc[-1]
                self.entry_price = current_price
                if signal == 'buy':
                    # Actualizar balance: se compra BTC con parte del capital
                    self.btc_balance = self.position_size
                    self.usd_balance = new_equity
                # Aquí podrías implementar la lógica para posiciones cortas si lo deseas.
            else:
                print(f"DEBUG: Valor de operación insuficiente incluso después de ajustes: {required_value:.2f}")



    def open_position(self, current_price, position_size):
        fee = position_size * current_price * commission
        cost = position_size * current_price + fee
        
        if cost > self.usd_balance:
            print("Fondos insuficientes incluso considerando equity total")
            return
        
        self.usd_balance -= cost
        self.btc_balance += position_size  # Añadir BTC al balance
        
        print(f"Compra realizada: {position_size:.6f} BTC a ${current_price:.2f}")
        print(f"Balance USD: ${self.usd_balance:.2f} | Balance BTC: {self.btc_balance:.6f}")

    def close_position(self, df, exit_price, reason):
        if self.btc_balance <= 0:
            print("Error: No hay BTC para vender")
            return
        
        if exit_price < df['low'].iloc[-1] or exit_price > df['high'].iloc[-1]:
            print(f"ADVERTENCIA: Precio de cierre {exit_price} fuera del rango de la vela (L:{df['low'].iloc[-1]}, H:{df['high'].iloc[-1]})")
        
        fee = self.btc_balance * exit_price * commission
        proceeds = (self.btc_balance * exit_price) - fee
        
        self.usd_balance += proceeds
        profit = proceeds - (self.btc_balance * self.entry_price)
        
        print(f"Venta realizada: {self.btc_balance:.6f} BTC a ${exit_price:.2f}")
        print(f"Profit: ${profit:.2f} | Balance USD: ${self.usd_balance:.2f}")
        
        self.trades.append({
            'entry': self.entry_price,
            'exit': exit_price,
            'profit': profit,
            'duration': (pd.Timestamp.now() - self.entry_time).total_seconds()/3600,
            'reason': reason
        })
        
        self.btc_balance = 0.0  # Resetear balance BTC

    def cross_validate(self, df):
        tscv = TimeSeriesSplit(n_splits=3)
        results = []
        for train_idx, test_idx in tscv.split(df):
            train = df.iloc[train_idx]
            test = df.iloc[test_idx]
            print(f"DEBUG: Validación cruzada - Tamaño Train: {len(train)}, Tamaño Test: {len(test)}")
            for i in range(len(test)):
                window = pd.concat([train, test.iloc[:i+1]])
                self.execute_trade(window)
            results.append(self.equity[-1])
            print(f"DEBUG: Resultado de validación para este split: {self.equity[-1]}")
        return np.mean(results)

    def calculate_performance(self, final_price):
        total_equity = self.get_total_equity(final_price)
        returns = (total_equity / capital_inicial - 1) * 100
        
        print("\n=== Rentabilidad Real ===")
        print(f"Capital Inicial: ${capital_inicial:.2f}")
        print(f"Capital Final: ${total_equity:.2f}")
        print(f"Retorno Total: {returns:.2f}%")
        # print(f"Sharpe Ratio: {self.calculate_sharpe_ratio():.2f}")

def main():
    df = pd.DataFrame(exchange.fetch_ohlcv(symbol, timeframe, limit=1000), 
                      columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    engine = TradingEngine()
    df = engine.calculate_features(df)
    
    # Validación cruzada
    validation_result = engine.cross_validate(df)
    print(f"Rentabilidad promedio validada: {(validation_result/capital_inicial-1)*100:.2f}%")
    
    # Ejecución final
    engine = TradingEngine()
    for i in range(len(df)):
        print(f"DEBUG: Procesando barra {i+1}/{len(df)} - Timestamp: {df.iloc[i]['timestamp']}")
        engine.execute_trade(df.iloc[:i+1])
    
    final_price = df['close'].iloc[-1]
    engine.calculate_performance(final_price)

if __name__ == '__main__':
    main()
