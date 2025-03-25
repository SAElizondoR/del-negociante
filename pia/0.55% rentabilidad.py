import os
import ccxt
import pandas as pd
import ta
from dotenv import load_dotenv

# Configuración optimizada
load_dotenv()
symbol = 'BTC/USDT'
timeframe = '1h'  # Balance entre ruido y oportunidades
capital_inicial = 20.0
risk_per_trade = 0.5  # 50% del capital por operación
position_min = 0.0001  # 0.0001 BTC (≈$6 con BTC en $60k)
commission = 0.00075  # Comisión con BNB (0.075%)
minimum_trade_value = 5  # Mínimo de Binance ($10)
trailing_stop = 0.01  # 1% de trailing stop
take_profit = 0.01

exchange = ccxt.binance({
    'apiKey': os.getenv('API_KEY'),
    'secret': os.getenv('API_SECRET'),
    'options': {'adjustForTimeDifference': True}
})

# Variables de estado global
current_position = None
entry_price = 0.0
highest_profit = 0.0
trades = []
equity_curve = []

def get_data():
    """Obtiene datos con EMA rápida y ADX para filtrar tendencias fuertes."""
    bars = exchange.fetch_ohlcv(symbol, timeframe, limit=300)
    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Indicadores clave
    df['ema_21'] = ta.trend.EMAIndicator(df['close'], 21).ema_indicator()
    df['ema_50'] = ta.trend.EMAIndicator(df['close'], 50).ema_indicator()
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], 14).rsi()
    df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], 14).adx()

    print("Datos obtenidos (últimas 5 filas):")
    print(df.tail())
    return df.dropna()

def generate_signal(df):
    """Genera señales con confirmación de ADX y divergencia de RSI."""
    last = df.iloc[-1]
    # Debug: Imprimir los indicadores clave
    print(f"DEBUG: Última fila -> EMA21: {last['ema_21']}, EMA50: {last['ema_50']}, RSI: {last['rsi']}, ADX: {last['adx']}")
    
    signal = None
    # Compra: EMA21 > EMA50, RSI < 35, ADX > 25 (tendencia fuerte)
    if last['ema_21'] > last['ema_50'] and last['rsi'] < 35 and last['adx'] > 25:
        signal = 'buy'
    # Venta: EMA21 < EMA50, RSI > 65, ADX > 25
    elif last['ema_21'] < last['ema_50'] and last['rsi'] > 65 and last['adx'] > 25:
        signal = 'sell'

    print(f"DEBUG: Señal generada: {signal}")
    return signal

def calculate_position_size(last_price):
    """Calcula el tamaño de la posición dinámicamente, considerando la comisión y el slippage."""
    balance = equity_curve[-1] if equity_curve else capital_inicial
    risk_amount = balance * risk_per_trade
    slippage = last_price * 0.0005  # 0.05% de slippage
    position_size = min(risk_amount / (last_price + slippage), balance / last_price)
    
    total_value = position_size * last_price
    print(f"DEBUG: Balance: {balance}, Last Price: {last_price}, Risk Amount: {risk_amount}, Calculado Position Size: {position_size}, Valor de operación: {total_value}")
    
    # Asegurarse de que el tamaño de la posición no sea menor que el mínimo requerido
    if total_value < minimum_trade_value:
        print(f"DEBUG: Operación ignorada, valor mínimo no alcanzado: {total_value}")
        return 0  # No ejecutar trade si el tamaño de la posición es demasiado pequeño
    
    return position_size

def execute_trade(signal, df):
    global current_position, entry_price, highest_profit

    last_close = df.iloc[-1]['close']
    balance = equity_curve[-1] if equity_curve else capital_inicial
    position_size = calculate_position_size(last_close)
    
    print(f"DEBUG: Ejecutando trade con señal: {signal}, current_position: {current_position}, entry_price: {entry_price}, last_close: {last_close}")
    
    # Cierre de posición (evaluar condiciones para salir)
    if current_position:
        # Actualizar el trailing stop
        current_profit_pct = last_close / entry_price - 1
        if current_profit_pct > highest_profit:
            highest_profit = current_profit_pct
            print(f"DEBUG: Actualizando highest_profit a: {highest_profit:.4f}")
        
        # Definir los umbrales de cierre para posición long y short
        if current_position == 'long':
            stop_loss_threshold = entry_price * (1 - trailing_stop)
            take_profit_threshold = entry_price * (1 + take_profit)
            print(f"DEBUG: Long -> Stop Loss: {stop_loss_threshold}, Take Profit: {take_profit_threshold}")
            condition_close = last_close <= stop_loss_threshold or last_close >= take_profit_threshold
        else:
            stop_loss_threshold = entry_price * (1 + trailing_stop)
            take_profit_threshold = entry_price * (1 - take_profit)
            print(f"DEBUG: Short -> Stop Loss: {stop_loss_threshold}, Take Profit: {take_profit_threshold}")
            condition_close = last_close >= stop_loss_threshold or last_close <= take_profit_threshold
        
        if condition_close:
            fee = abs(last_close * position_size * commission)
            profit = (last_close - entry_price) * position_size if current_position == 'long' else (entry_price - last_close) * position_size
            balance += profit - fee
            trades.append({
                'type': 'sell' if current_position == 'long' else 'buy',
                'entry': entry_price,
                'exit': last_close,
                'profit': profit,
                'fee': fee
            })
            print(f"DEBUG: Posición cerrada: {current_position} a {last_close}. Profit: {profit:.4f}, Fee: {fee:.4f}, Nuevo balance: {balance:.4f}")
            current_position = None
            equity_curve.append(balance)
        else:
            print(f"DEBUG: Condiciones para cierre no cumplidas. Last_close: {last_close}, Entry: {entry_price}, Current Profit: {current_profit_pct:.4f}")
    
    # Apertura de posición si no hay posición abierta y hay señal
    if not current_position and signal:
        if position_size == 0:
            print("DEBUG: Tamaño de posición 0, no se abre operación.")
            return

        fee = position_size * last_close * commission
        entry_price = last_close
        current_position = 'long' if signal == 'buy' else 'short'
        highest_profit = 0.0

        # En lugar de descontar el valor total de la operación, solo se descuenta la comisión inicial
        new_balance = balance - fee
        equity_curve.append(new_balance)
        print(f"DEBUG: Posición abierta: {current_position} a {entry_price}, Fee: {fee:.4f}, Balance actualizado: {new_balance:.4f}")

def run_strategy():
    df = get_data()
    if len(df) < 100:
        print("DEBUG: Datos insuficientes")
        return

    equity_curve.append(capital_inicial)
    print(f"DEBUG: Capital inicial: {capital_inicial}")
    
    for i in range(100, len(df)):
        window = df.iloc[:i]
        print(f"\nDEBUG: Procesando barra {i} / {len(df)} - Timestamp: {window.iloc[-1]['timestamp']}")
        signal = generate_signal(window)
        execute_trade(signal, window)

        # Forzar cierre si la tendencia se revierte
        if current_position:
            if current_position == 'long':
                condition_revert = window.iloc[-1]['ema_21'] < window.iloc[-1]['ema_50'] * 0.99
            else:
                condition_revert = window.iloc[-1]['ema_21'] > window.iloc[-1]['ema_50'] * 1.01
            if condition_revert:
                print(f"DEBUG: Tendencia revertida detectada para {current_position}. Forzando cierre de posición.")
                execute_trade('sell' if current_position == 'long' else 'buy', window)
    
    # Métricas finales
    total_trades = len(trades)
    winning_trades = len([t for t in trades if t['profit'] > 0])
    profit_factor = (sum(t['profit'] for t in trades if t['profit'] > 0) / 
                     abs(sum(t['profit'] for t in trades if t['profit'] < 0))
                     ) if total_trades else 0

    print("\n=== Resultados ===")
    print(f"Capital final: ${equity_curve[-1]:.2f}")
    print(f"Rentabilidad: {(equity_curve[-1]/capital_inicial-1)*100:.2f}%")
    print(f"Operaciones: {total_trades} | Win Rate: {winning_trades/total_trades*100 if total_trades else 0:.1f}%")
    print(f"Profit Factor: {profit_factor:.2f}")

if __name__ == '__main__':
    run_strategy()
