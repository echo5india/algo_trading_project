import datetime
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import time
import sys
import copy
import math

from simtools import log_message

# Lee-Ready tick strategy simulator

# Record a trade in our trade array
def record_trade( trade_df, idx, trade_px, trade_qty, current_bar, trade_type, side ):
    print( "Trade! {} {} {} {}".format( idx, trade_px, trade_qty, current_bar ) )
    trade_df.loc[ idx ] = [ trade_px, trade_qty, current_bar, trade_type, side ]
    return

# Record a trade in our trade array
def record_candle_trades( trade_candlesticks, idx, trade_qty, side):
    print(trade_qty)
    if pd.isna(trade_candlesticks['shares'][idx]):
        trade_candlesticks['shares'][idx] = trade_qty
    else:
        trade_candlesticks['shares'][idx] += trade_qty
    trade_candlesticks['side'][idx] = side
    trade_candlesticks['filled'][idx] = 0
    return

def record_candlesticks(candlesticks_df,idx,open_,close_,high_,low_):
    candlesticks_df['open'][idx] = open_
    candlesticks_df['close'][idx] = close_
    candlesticks_df['high'][idx] = high_
    candlesticks_df['low'][idx] = low_
    return

def record_candlestick_indicators(candlestick_indicators,idx,indicator_list,trade):
    y = np.append(indicator_list,trade)
    candlestick_indicators.loc[idx] = y
    return

def trade_statistics(trade_df,net_position,final_bid,final_ask, midpoints, ask_values, bid_values):
    #Total P&L (dollar amount)
    sell_trades = trade_df[trade_df['side']=='s'][['price','shares']]
    buy_trades = trade_df[trade_df['side']=='b'][['price','shares']]
    sell_sum = (sell_trades['price']*sell_trades['shares']).sum() #Dollar amount of all sales
    buy_sum = (buy_trades['price']*buy_trades['shares']).sum() #Dollar amount of all buys
    
    # Any outstanding amount we have at maturity is sold at bid or bought at ask
    total_pnl = sell_sum - buy_sum + (net_position>0) * net_position * final_bid + \
                (net_position < 0) * net_position * final_ask

    # Calculate intraday P&L (time series). P&L has two components. Roughly:
    #       1. realized "round trip" P&L  sum of (sell price - buy price) * shares traded
    #       2. unrealized P&L of open position:  quantity held * (current price - avg price)
    midpoints = midpoints.loc[trade_df.index]
    ask_values = ask_values.loc[trade_df.index]
    bid_values = bid_values.loc[trade_df.index]
    intra_pnl_df = trade_df[['price','shares','side']].copy()
    intra_pnl_df.loc[intra_pnl_df['side'] =='s', 'side'] = 1
    intra_pnl_df.loc[intra_pnl_df['side'] =='b', 'side'] = -1
    intra_pnl_df['dollar_amt'] = intra_pnl_df.prod(axis=1)
    intra_pnl_df['signed_shares'] = -intra_pnl_df['side'] * intra_pnl_df['shares']
    intra_pnl_df['cum_signed_shares'] = intra_pnl_df['signed_shares'].cumsum(0)
    intra_pnl_df['midpoint'] = midpoints
    intra_pnl_df['cum_pnl'] = intra_pnl_df['dollar_amt'].cumsum(0) + \
                                (intra_pnl_df['cum_signed_shares'] > 0) * \
                                bid_values * intra_pnl_df['cum_signed_shares'] + \
                                (intra_pnl_df['cum_signed_shares'] <= 0) * ask_values * \
                                intra_pnl_df['cum_signed_shares']
    intra_pnl_df['intra_pnl'] = intra_pnl_df['cum_pnl'] - intra_pnl_df['cum_pnl'].shift(1)
    intra_pnl_df['qty_behind'] = (intra_pnl_df['shares']*(-intra_pnl_df['side'])).cumsum(0)
    
    #Calculate worst and best intraday P&L
    worst_intra = intra_pnl_df['intra_pnl'].dropna().min()
    best_intra = intra_pnl_df['intra_pnl'].dropna().max()
    
    #calculate maximum position (both long and short)
    max_long = max(intra_pnl_df['cum_signed_shares'])
    max_short = min(intra_pnl_df['cum_signed_shares'])
    
    # Maximum and minimum  PnL over the day
    min_pnl = intra_pnl_df['cum_pnl'].dropna().min()
    max_pnl = intra_pnl_df['cum_pnl'].dropna().max()
    
    # Maximum and minimum buy orders
    min_shares_long = intra_pnl_df['signed_shares'][intra_pnl_df['signed_shares'] >= 0].min()
    max_shares_long = intra_pnl_df['signed_shares'][intra_pnl_df['signed_shares'] >= 0].max()
    
    # Maximum and mininum sell orders
    min_shares_short = intra_pnl_df['signed_shares'][intra_pnl_df['signed_shares'] <= 0].min()
    max_shares_short = intra_pnl_df['signed_shares'][intra_pnl_df['signed_shares'] <= 0].max()
    
    return total_pnl, intra_pnl_df, worst_intra, best_intra, max_long, max_short, min_pnl, max_pnl, min_shares_long, max_shares_long, min_shares_short, max_shares_short



# Get next order quantity
def calc_order_quantity(raw_order_qty, midpoint, fair_value, order_side, quant_coef):
    alpha = quant_coef
    if order_side == 'b':
        return round(alpha*((fair_value-midpoint))/((midpoint)))
        # We experienced with other methods of calculating the quantity, making the difference from the midpoint more important
        #return round(alpha*((fair_value-midpoint)/(midpoint))**(1/2))
    else:
        return round(-alpha*((fair_value-midpoint))/((midpoint)))
        # We experienced with other methods of calculating the quantity, making the difference from the midpoint more important
        #return round(alpha*((fair_value-midpoint)/(midpoint**2))**(1/2))
        

    
# MAIN ALGO LOOP
def algo_loop( trading_day, factors_included = [1, 3, 4, 5], coef_included = [2, 8e-4, 10, 10], limit_orders_included = True, quantity_coef_=4000000, display_factors = False ):
    
    factors_included = np.array(factors_included)
    if len(factors_included) != len(coef_included):
        print('Factors included need to be the same length as coefficients included')
        return 0
    
    log_message( 'Beginning Tick Strategy run' )
    
    # Average spread is calculated with the NBBO prices
    avg_spread = ( trading_day[(( trading_day.qu_source == 'N' ) & ( trading_day.natbbo_ind == 4 )) | ((trading_day.qu_source == 'C') & (trading_day.natbbo_ind == 'G'))].ask_px - trading_day[(( trading_day.qu_source == 'N' ) & ( trading_day.natbbo_ind == 4 )) | ((trading_day.qu_source == 'C') & (trading_day.natbbo_ind == 'G'))].bid_px ).mean()
    # Calculating half the spread (used in factors)
    half_spread = avg_spread / 2
    print( "Average stock spread for sample: {:.4f}".format(avg_spread) )

    # init our price and volume variables
    [ last_price, last_size, bid_price, bid_size, ask_price, ask_size, volume ] = np.zeros(7)

    # init our counters
    [ trade_count, quote_count, cumulative_volume ] = [ 0, 0, 0 ]
    
    # init some time series objects for collection of telemetry
    fair_values = pd.Series(index = trading_day.index)
    midpoints = pd.Series(index = trading_day.index)
    bid_values = pd.Series(index = trading_day.index)
    ask_values = pd.Series(index = trading_day.index)
    tick_factors = pd.Series(index = trading_day.index)
    risk_factors = pd.Series(index = trading_day.index) #-Added risk factor series - Matt
    c1_factors = pd.Series(index = trading_day.index)
    c2_factors = pd.Series(index = trading_day.index)
    c3_factors = pd.Series(index = trading_day.index)
    c4_factors = pd.Series(index = trading_day.index)
    c5_factors = pd.Series(index = trading_day.index)   
    ma_factors = pd.Series(index = trading_day.index)
    indicator_list = np.zeros(5)

    # let's set up a container to hold trades. preinitialize with the index
    trades = pd.DataFrame( columns = [ 'price' , 'shares', 'bar', 'trade_type', 'side' ], index=trading_day.index )
    
    ma_df = pd.DataFrame(columns = ['MA_LT', 'MA_ST', 'MA_LT_m', 'MA_ST_m'], index = trading_day.index)
    # Try to make this more general for many days
    trades_candle = pd.DataFrame( columns = [ 'shares', 'side' , 'filled'], 
                    index = [datetime.strptime(str(trading_day.index[0])[:10] + ' 09:30:00',
                    '%Y-%m-%d %H:%M:%S') + timedelta(minutes=x) for x in range(391)])
    candlesticks_df = pd.DataFrame( columns = [ 'open', 'close' , 'high','low'], 
                    index = [datetime.strptime(str(trading_day.index[0])[:10] + ' 09:30:00',
                    '%Y-%m-%d %H:%M:%S') + timedelta(minutes=x) for x in range(391)])
    candlestick_indicators = pd.DataFrame( columns = [4,5,6,7,8]+['trades'], 
                index = [datetime.strptime(str(trading_day.index[0])[:10] + ' 09:30:00',
                '%Y-%m-%d %H:%M:%S') + timedelta(minutes=x) for x in range(391)])

    # MAIN EVENT LOOP
    current_bar = 0
    net_position = 0

    # track state and values for a current working order
    live_order = False
    live_order_price = 0.0
    live_order_quantity = 0.0
    order_side = '-'
    bid_price = 0
    ask_price = 0

    # other order and market variables
    total_quantity_filled = 0
    vwap_numerator = 0.0
    total_trade_count = 0
    total_agg_count = 0
    total_pass_count = 0

    # fair value pricing variables
    midpoint = 0.0
    fair_value = 0.0
    message_type = 0 
    
    # Moving average
    ma_factor = 0.0
    if 1 in factors_included:
        ma_coef = coef_included[np.where(factors_included == 1)[0][0]]#2#0.1
        temp = trading_day[trading_day.trade_px.isna() == False].shape[0]
        ma_interval = math.floor(temp*5/390)
        ma_interval_long = math.floor(temp*20/390)
        ma_vector = np.zeros(ma_interval)
        ma_vector_long = np.zeros(ma_interval_long)
        ma_vector_plot = np.zeros(ma_interval)
        ma_vector_long_plot = np.zeros(ma_interval_long)
        ma_i = 0
        ma_i_long = 0
        ma_bool = True
    else:
        ma_coef = 0

    # define our accumulator for the tick EMA
    tick_factor = 0.0
    if 2 in factors_included:
        tick_coef = coef_included[np.where(factors_included == 2)[0][0]]#1.3
        tick_window = 30
        tick_ema_alpha = 2 / (tick_window + 1)
        prev_tick = 0
        prev_price = 0
    else:
        tick_coef = 0.0
    
    # risk factor for part 2
    risk_factor = 0.0
    if 3 in factors_included:
        risk_coef = coef_included[np.where(factors_included == 3)[0][0]]#8e-4
    else:
        risk_coef = 0
    
    # Candlestick factor 1
    c1_factor = 0.0
    if 4 in factors_included:
        c1_coef = coef_included[np.where(factors_included == 4)[0][0]]#10 #1
        c1_shift = 5
    else:
        c1_coef = 0
        c1_shift = 0
    
    # Candlestick factor 2
    c2_factor = 0.0
    if 5 in factors_included:
        c2_coef = coef_included[np.where(factors_included == 5)[0][0]]#10
        c2_shift = 10
    else:
        c2_coef = 0
        c2_shift = 0
        
    # Candlestick factor 3
    c3_factor = 0.0
    if 6 in factors_included:
        c3_coef = coef_included[np.where(factors_included == 6)[0][0]]# 10
        c3_shift = 60
    else:
        c3_coef = 0.0
        c3_shift = 0.0
    
    # Candlestick factor 4
    c4_factor = 0.0
    if 7 in factors_included:
        c4_coef = coef_included[np.where(factors_included == 7)[0][0]]
        c4_shift = 30
    else:
        c4_coef = 0.0
        c4_shift = 0
    
    # Candlestick factor 5
    c5_factor = 0.0
    if 8 in factors_included:
        c5_coef = coef_included[np.where(factors_included == 8)[0][0]]
        c5_shift = 30
    else:
        c5_coef = 0.0
        c5_shift = 0
        
        
    current_minute = 0
    last_minute = 0
    last_high = 0
    last_low = 0
    last_open = 0
    last_close = 0
    current_high = 0
    current_low = 0
    current_open = 0
    current_close = 0
    first_minute = True

    
    # factors calculating the quantity to buy and sell
    quantity_coef = quantity_coef_#4000000
    # Floor on the minimum amount we need to place a limit order
    min_lim_order = 50
    
    # Variable that takes care of the fact that a passive order and aggressive order can be completed at the same time.
    # Instead of overwriting the trade in trades_df, we only allow one trade each iteration
    live_order_executed = False
    candlestick_calculation = False
    
    log_message( 'starting main loop' )
    for index, row in trading_day.iterrows():
        
        last_minute = current_minute
        current_minute = index.minute
        # Change the if sentence if we have more than 1 day of data # IT WORKS FOR MANY DAYS
        if current_minute > last_minute or (current_minute == 0 and last_minute == 59) or first_minute:
            if first_minute:
                first_minute = not first_minute
            else:
                candlestick_calculation = True
                last_high = current_high
                last_low = current_low
                last_open = current_open
                last_close = current_close

                if current_minute == 0:
                    last_hour = index.hour - 1
                else:
                    last_hour = index.hour
                
                current_minute_data = trading_day[(trading_day.index.hour == last_hour) & 
                                                  (trading_day.index.minute == last_minute) & 
                                                  (trading_day.trade_px.isna() == False)]
                
                temp = current_minute_data['trade_px']
                if len(temp) == 0:
                    current_high = last_close
                    current_low = last_close
                    current_open = last_close
                    current_close = last_close
                else:
                    current_high = temp.max()
                    current_low = temp.min()
                    current_open = temp[0]
                    current_close = temp[-1]
                
        # get the time of this message
        time_from_open = (index - pd.Timedelta( hours = 9, minutes = 30 ))
        minutes_from_open = (time_from_open.hour * 60) + time_from_open.minute
        
        ## Have to fix this because if a trade comes in, it has to fill the limit order as well
        ## but not just perform the candlestick trade and nothing more
        if candlestick_calculation:
            new_index = str(index)[:10] + ' ' + str(index.hour).zfill(2) + ':' + \
                        str(index.minute).zfill(2) + ':00'

            record_candlesticks(candlesticks_df,new_index,current_open, 
                                current_close, current_high, current_low)
            if trades_candle['shares'][(trades_candle.index <= new_index) & 
                                       (trades_candle['filled'] == 0)].sum()!=0:
                live_order_executed = True
                if trades_candle['shares'][(trades_candle.index <= new_index) & 
                                       (trades_candle['filled'] == 0)].sum()>0:
                    print('Closing Position')
                    temp_side = 's'
                    record_trade( trades, index, bid_price, trades_candle['shares'][ 
                                        (trades_candle.index <= new_index) &                                                                          (trades_candle['filled'] == 0)].sum(),
                                         current_bar, 'a', temp_side )
                    net_position = net_position - trades_candle['shares'][ 
                                        (trades_candle.index <= new_index) &                                                                         (trades_candle['filled'] == 0)].sum()
                    total_quantity_filled += trades_candle['shares'][ 
                                        (trades_candle.index <= new_index) &                                                                          (trades_candle['filled'] == 0)].sum()
                    total_agg_count += 1
                    trades_candle['filled'][(trades_candle.index <= new_index) &                                                                          (trades_candle['filled'] == 0)] = 1
                else:
                    print('Closing Position')
                    temp_side = 'b'
                    record_trade( trades, index, ask_price, -trades_candle['shares'][ 
                                        (trades_candle.index <= new_index) &                                                                        (trades_candle['filled'] == 0)].sum(),
                                         current_bar, 'a', temp_side )
                    net_position = net_position - trades_candle['shares'][ 
                                        (trades_candle.index <= new_index) &                                                                          (trades_candle['filled'] == 0)].sum()
                    total_quantity_filled -= trades_candle['shares'][ 
                                        (trades_candle.index <= new_index) &                                                                          (trades_candle['filled'] == 0)].sum()
                    total_agg_count += 1
                    trades_candle['filled'][(trades_candle.index <= new_index) &                                                                          (trades_candle['filled'] == 0)] = 1
        
        
        # MARKET DATA HANDLING
        if pd.isna( row.trade_px ): # it's a quote
            # skip if not NBBO
            if not ((( row.qu_source == 'N') and (row.natbbo_ind == 4)) or 
                    ((row.qu_source == 'C') and (row.natbbo_ind == 'G'))):
                candlestick_calculation = False
                bid_values[index] = bid_price
                ask_values[index] = ask_price
                continue
            # set our local NBBO variables
            if ( row.bid_px > 0 and row.bid_size > 0 ):
                bid_price = row.bid_px
                bid_size = row.bid_size
            if ( row.ask_px > 0 and row.ask_size > 0 ):
                ask_price = row.ask_px
                ask_size = row.ask_size
            quote_count += 1
            message_type = 'q'
        else: # it's a trade
            # store the last trade price
            prev_price = last_price
            # now get the new data
            last_price = row.trade_px
            last_size = row.trade_size
            trade_count += 1
            cumulative_volume += row.trade_size
            vwap_numerator += last_size * last_price
            message_type = 't'
            
            # CHECK OPEN ORDER(S) if we have a live order, 
            # has it been filled by the trade that just happened?
            if live_order and not live_order_executed and limit_orders_included:
                if ( order_side == 'b' ) and ( last_price <= live_order_price ) : 
                    #Our side hasn't updated because once it does, the live order is cancelled
                    print('Limit buy @ %.3f' % (live_order_price))
                    fill_size = min( live_order_quantity, last_size )
                    record_trade( trades, index, live_order_price, fill_size, current_bar, 'p', order_side )
                    net_position = net_position + fill_size
                    total_quantity_filled += fill_size
                    total_pass_count += 1

                    # We chose to not cancel the limit order once a part of it was filled.
                    # We update the quantity
                    if fill_size >= live_order_quantity:
                        live_order = False
                        live_order_price = 0.0
                        live_order_quantity = 0.0
                    else:
                        live_order_quantity -= fill_size
                    
                    live_order_executed = True
                    

                if ( order_side == 's' ) and ( last_price >= live_order_price ) :
                    print('Limit sell @ %.3f' % (live_order_price))
                    fill_size = min( live_order_quantity, last_size )
                    record_trade( trades, index, live_order_price, fill_size, current_bar, 'p', order_side )
                    net_position = net_position - fill_size
                    total_quantity_filled += fill_size
                    total_pass_count += 1

                    # We chose to not cancel the limit order once a part of it was filled. 
                    # We update the quantity
                    if fill_size >= live_order_quantity:
                        live_order = False
                        live_order_price = 0.0
                        live_order_quantity = 0.0
                    else:
                        live_order_quantity -= fill_size
                                        
                    live_order_executed = True
        
        # MOVING AVERAGE
        # calc moving average factor
        if 1 in factors_included:
            if message_type == 't' and midpoint > 0:
                #ma_spread = last_spread/2
                #if order_side == 'b':
                ma_vector[ma_i] = last_price #- midpoint#ask_price - ma_spread
                ma_vector_long[ma_i_long] = last_price #- midpoint
                ma_vector_plot[ma_i] = last_price
                ma_vector_long_plot[ma_i_long] = last_price
                #else:
                #    ma_vector[ma_i] = bid_price + ma_spread
                if ma_i == ma_interval-1:
                    ma_i = 0
                else:
                    ma_i += 1
                
                if ma_i_long == ma_interval_long-1:
                    ma_i_long = 0
                else:
                    ma_i_long += 1
                
                #i = (i+1)*((ma_interval - 1) != i)
                if ma_vector[-1] != 0 and ma_vector_long[-1] != 0:
                    st_mean = np.mean(ma_vector)
                    lt_mean = np.mean(ma_vector_long)
                    ma_df['MA_ST'][index] = np.mean(ma_vector_plot)
                    ma_df['MA_LT'][index] = np.mean(ma_vector_long_plot)
                    ma_df['MA_ST_m'][index] = st_mean
                    ma_df['MA_LT_m'][index] = lt_mean
                    if st_mean > lt_mean and not ma_bool:
                        ma_factor = 1
                        ma_bool = not ma_bool
                    elif st_mean < lt_mean and ma_bool:
                        ma_factor = -1
                        ma_bool = not ma_bool
                    else:
                        ma_factor = 0
                    #ma_factor = st_mean - lt_mean
                else:
                    ma_factor = 0
                
        # TICK FACTOR
        # only update if it's a trade
        if 2 in factors_included:
            if message_type == 't':
                # calc the tick
                this_tick = np.sign(last_price - prev_price)
                if this_tick == 0:
                    this_tick = prev_tick

                # now calc the tick
                if tick_factor == 0:
                    tick_factor = this_tick
                else:
                    tick_factor = ( tick_ema_alpha * this_tick ) + ( 1 - tick_ema_alpha ) * tick_factor    

                # store the last tick
                prev_tick = this_tick
            
        # RISK FACTOR
        # calc the risk factor
        if 3 in factors_included:
            if net_position < 0: #Need to buy back shares
                # Risk factor is dependent on how short we are
                risk_factor = -net_position 
            elif net_position > 0:
                # Risk factor is dependent on how long we are
                risk_factor = -net_position
            else:
                # if the net position is 0, we cancel the risk factor
                risk_factor = 0

        # CANDLESTICK 1 FACTOR
        # DARKCLOUD COVER
        if 4 in factors_included:
            c1_factor = 0
            if candlestick_calculation:
                if ((last_close > last_open) and
                (((last_close + last_open) / 2) > current_close) and
                (current_open > current_close) and
                (current_open > last_close) and
                (current_close > last_open) and
                ((current_open - current_close) / (.001 + (current_high - current_low)) > 0.6)):
                    c1_factor = 1#(abs(last_close - current_close) + abs(current_close - current_open))/2
                    indicator_list[0] = c1_factor
                    
        # CANDLESTICK 2 FACTOR
        # DOJI
        if 5 in factors_included:
            c2_factor = 0
            if candlestick_calculation:
                if abs(current_close - current_open) / (current_high - current_low) < 0.1 and \
               (current_high - max(current_close, current_open)) > (3 * abs(current_close - current_open)) and \
               (min(current_close, current_open) - current_low) > (3 * abs(current_close - current_open)):
                    c2_factor = 1#(abs(last_close - current_close) + abs(current_close - current_open))/2
                    indicator_list[1] = c2_factor
                    
        # CANDLESTICK 3 FACTOR
        # BEARISH ENGULFING
        if 6 in factors_included:
            c3_factor = 0
            if candlestick_calculation:
                if (current_open >= last_close > last_open and
                current_open > current_close and
                last_open >= current_close and 
                current_open - current_close > last_close - last_open):
                    c3_factor = 1
                    indicator_list[2] = c3_factor
                
        # CANDLESTICK 4 FACTOR
        # HAMMER
        if 7 in factors_included:
            c4_factor = 0
            if candlestick_calculation:
                if (((current_high - current_low) > 3 * (current_open - current_close)) and
                ((current_close - current_low) / (.001 + current_high - current_low) > 0.6) and
                ((current_open - current_low) / (.001 + current_high - current_low) > 0.6)):
                    c4_factor = -1
                    indicator_list[3] = c4_factor
        
        # CANDLESTICK 5 FACTOR
        # INVERTED HAMMER
        if 8 in factors_included:
            c5_factor = 0
            if candlestick_calculation:
                if (((current_high - current_low) > 3 * (current_open - current_close)) and
                ((current_high - current_close) / (.001 + current_high - current_low) > 0.6)
                and ((current_high - current_open) / (.001 + current_high - current_low) > 0.6)):
                    c5_factor = -1
                    indicator_list[4] = c5_factor
        
        
        # PRICING LOGIC
        new_midpoint = bid_price + ( ask_price - bid_price ) / 2
        if new_midpoint > 0:
            midpoint = new_midpoint


        # FAIR VALUE CALCULATION
        # check inputs, skip of the midpoint is zero, we've got bogus data (or we're at start of day)
        if midpoint == 0:
            print( "{} no midpoint. b:{} a:{}".format( index, bid_price, ask_price ) )
            continue
        fair_value = midpoint + half_spread * (( risk_coef * risk_factor ) +
                    ( ma_coef * ma_factor ) + ( c1_coef * c1_factor ) + 
                    ( c2_coef * c2_factor ) + ( c3_coef * c3_factor ) +
                    ( c4_coef * c4_factor ) + ( c5_coef * c5_factor ))
        
        # This prints out all important information. Good to see how the factors are functioning
        if display_factors: 
            print('fair bid ask mid tick risk')
            print('%.3f %.3f %.3f %.3f %.3f %.3f' % ((fair_value), (bid_price), 
            (ask_price), (midpoint), (half_spread * ( tick_coef * tick_factor )),
            (half_spread* (risk_coef * risk_factor))))
        
        # collect our data
        fair_values[ index ] = fair_value
        midpoints[ index ] = midpoint
        bid_values[ index ] = bid_price
        ask_values[ index ] = ask_price
        tick_factors[ index ] = half_spread * (( tick_coef * tick_factor ))
        risk_factors[index] = half_spread * ( risk_coef * risk_factor ) #Added risk_factor to series
        c1_factors[index] = c1_factor * c1_coef * half_spread
        c2_factors[index] = c2_factor * c2_coef * half_spread
        c3_factors[index] = c3_factor * c3_coef * half_spread
        c4_factors[index] = c4_factor * c4_coef * half_spread
        c5_factors[index] = c5_factor * c5_coef * half_spread
        ma_factors[index] = half_spread * ma_coef * ma_factor
        
        if candlestick_calculation:
            record_candlestick_indicators(candlestick_indicators,new_index,np.abs(indicator_list),trade=0)

        # TRADE DECISION
        # We want to sell if the fair value is below the midpoint and buy if the fair value is above
        if fair_value < midpoint:
            order_side = 's'
        elif fair_value > midpoint:
            order_side = 'b'
        else:
            order_side = '-'
        
        
        # TRADING LOGIC
        # check where our FV is versus the BBO and constrain
        if order_side == 'b':
            # Don't trade if a limit order was executed in the same iteration
            if (fair_value >= ask_price) and (live_order_executed == False): 
                new_trade_price = ask_price

                # now place our aggressive order: assume you can execute the full size across spread
                new_order_quantity = calc_order_quantity(net_position, midpoint, 
                                                         fair_value, order_side, quantity_coef)
                new_order_quantity = np.minimum(new_order_quantity, 2000)
                
                # If the new order quantity is above 0, we do a trade
                if new_order_quantity != 0:
                    total_agg_count += 1
                    hour_index = index.hour
                    if np.any(np.array([c1_factor, c2_factor, c3_factor, c4_factor, c5_factor]) > 0):
                        # sum([abs(x) for x in [c1_factor,c2_factor,c3_factor,c4_factor,c5_factor]]) != 0:
                        if c1_factor > 0:
                            time_shift = c1_shift
                        elif c2_factor > 0:
                            time_shift = c2_shift
                        elif c3_factor > 0:
                            time_shift = c3_shift
                        elif c4_factor > 0:
                            time_shift = c4_shift
                        else:
                            time_shift = c5_shift

                        if index.minute + time_shift > 59: #Only works for small time shifts
                            hour_index = index.hour + 1
                            minute_index = index.minute + time_shift - 60
                        else:
                            hour_index = index.hour
                            minute_index = index.minute + time_shift
                        if hour_index <= 15:
                            record_candlestick_indicators(candlestick_indicators,
                                                          new_index,indicator_list,trade=1)
                            new_index = str(trading_day.index[0])[:10] + \
                                        ' ' + str(hour_index).zfill(2) + \
                                        ':' + str(minute_index).zfill(2) + ':00'
                            record_candle_trades( trades_candle, new_index, new_order_quantity, order_side )
                    if hour_index <= 15:
                        record_trade( trades, index, new_trade_price, new_order_quantity, 
                                      current_bar, 'a', order_side )
                        net_position = net_position + new_order_quantity

                        # Good to visualize the factor values by printing out during a trade
                        if display_factors:
                            print('Fair value: ', fair_value)
                            print('Tick: ', half_spread * ( ( tick_coef * tick_factor )))
                            print('Risk: ', half_spread * ( risk_coef * risk_factor ))

                        # update quantity remaining
                        total_quantity_filled += new_order_quantity

                        live_order_quantity = 0.0
                        live_order_price = 0.0
                        live_order = False

            elif limit_orders_included: # we're not yet willing to cross the spread, stay passive
                order_quantity = calc_order_quantity(net_position, midpoint,
                                                     fair_value, order_side, quantity_coef)
                
                # Cap our orders to 2000 shares.
                order_quantity = np.minimum(order_quantity, 2000)
                
                # Don't send a limit order if it's too small (to minimize transaction costs)
                if order_quantity >= min_lim_order:
                    live_order_price = bid_price
                    live_order_quantity = order_quantity
                    live_order = True


        elif order_side == 's':
            # Don't trade if a limit order was executed in the same iteration
            if (fair_value <= bid_price) and (live_order_executed == False):
                new_trade_price = bid_price
                # now place our aggressive order: assume you can execute the full size across spread
                new_order_quantity = calc_order_quantity(net_position, midpoint,
                                                         fair_value, order_side, quantity_coef)
                new_order_quantity = np.minimum(new_order_quantity, 2000)
                
                if new_order_quantity != 0:
                    total_agg_count += 1
                    hour_index = index.hour
                    hour_index = index.hour
                    if np.any(np.array([c1_factor, c2_factor, c3_factor, c4_factor, c5_factor]) < 0):
                        if c1_factor < 0:
                            time_shift = c1_shift
                        elif c2_factor < 0:
                            time_shift = c2_shift
                        elif c3_factor < 0:
                            time_shift = c3_shift
                        elif c4_factor < 0:
                            time_shift = c4_shift
                        else:
                            time_shift = c5_shift

                        if index.minute + time_shift > 59: #Only works for small time shifts
                            hour_index = index.hour + 1
                            minute_index = index.minute + time_shift - 60
                        else:
                            hour_index = index.hour
                            minute_index = index.minute + time_shift
                        if hour_index <= 15:
                            record_candlestick_indicators(candlestick_indicators,new_index,
                                                          np.abs(indicator_list),trade=-1)
                            new_index = str(trading_day.index[0])[:10] + \
                                        ' ' + str(hour_index).zfill(2) + \
                                        ':' + str(minute_index).zfill(2) + ':00'
                            record_candle_trades( trades_candle, new_index, -new_order_quantity, order_side )
                if hour_index <= 15:
                    record_trade( trades, index, new_trade_price, new_order_quantity, 
                                            current_bar, 'a', order_side )
                    net_position = net_position - new_order_quantity

                    # update quantity remaining
                    total_quantity_filled += new_order_quantity

                    # Print factors for convenience
                    if display_factors:
                        print('Fair value: ', fair_value)
                        print('Tick: ', half_spread * ( ( tick_coef * tick_factor )))
                        print('Risk: ', half_spread * ( risk_coef * risk_factor ))

                    live_order_quantity = 0.0
                    live_order_price = 0.0
                    live_order = False

                
            elif limit_orders_included: # not yet willing to cross spread
                order_quantity = calc_order_quantity(net_position, midpoint, 
                                                        fair_value, order_side, quantity_coef)
                
                # Cap or order quantity to 2000 shares each time
                order_quantity = np.minimum(order_quantity, 2000)
                
                # Don't send a limit order if it's too small (to minimize transaction costs)
                if order_quantity >= min_lim_order:
                    live_order_price = ask_price
                    live_order_quantity = order_quantity
                    live_order = True
                    
        else:
            pass
            # no order here. for now just continue

        live_order_executed = False
        candlestick_calculation = False
        indicator_list = np.zeros(5)

    # looping done
    log_message( 'end simulation loop' )
    log_message( 'order analytics' )

    # Now, let's look at some stats
    trades = trades.dropna()
    day_vwap = vwap_numerator / cumulative_volume
    
    # prep our text output
    avg_price = 0

    if trades[ 'shares' ].sum() != 0:
        avg_price = (trades[ 'price' ] * trades[ 'shares' ]).sum() / trades[ 'shares' ].sum()

    log_message( 'Algo run complete.' )
    
    #print(midpoints.loc['2019-10-10 15:57:00.020201161'])
    [total_pnl, intra_pnl_df, worst_intra,
    best_intra, max_long, max_short, min_pnl, max_pnl,
    min_shares_long, max_shares_long, min_shares_short, max_shares_short] = trade_statistics(trades,
                                net_position,bid_price,ask_price,midpoints,ask_values,bid_values)
    
    statistics = [total_pnl, intra_pnl_df, worst_intra,best_intra, max_long, max_short, min_pnl,
                  max_pnl,min_shares_long, max_shares_long, min_shares_short, max_shares_short]

    # assemble results and return
    return { 'midpoints' : midpoints,
             'fair_values' : fair_values,
             'tick_factors' : tick_factors,
             'risk_factors' : risk_factors,
             'c1_factors' : c1_factors,
             'c2_factors' : c2_factors,
             'c3_factors' : c3_factors,
             'c4_factors' : c4_factors,
             'c5_factors' : c5_factors,
             'ma_factors' : ma_factors,
             'trades' : trades,
             'quote_count' : quote_count,
             'day_vwap' : day_vwap,
             'avg_price' : avg_price,
             'net_position': net_position,
             'last_bid' : bid_price,
             'last_ask' : ask_price,
             'midpoints' : midpoints,
             'ask_values' : ask_values,
             'bid_values' : bid_values,
             'statistics' : statistics,
             'candlesticks' : candlesticks_df,
             'candlestick_indicators' : candlestick_indicators,
             'ma_df' : ma_df
           }