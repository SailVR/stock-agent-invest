import tushare as ts

pro = ts.pro_api('7a37c632fd0bea6ef7c085f4f3496237f32b54c31481f3bf5dcc5e1d')

#获取单日全部股票数据
df = pro.moneyflow(trade_date='20260615')

#获取单个股票数据
df = pro.moneyflow(ts_code='002149.SZ', start_date='20260615', end_date='20260615')

print(df)

#获取2026年3月12日的当日所有个股异常信息
df = pro.stk_shock(trade_date='20260416')

#获取股票”协鑫能科“2025年以来每个交易日的个股异常信息
df = pro.stk_shock(ts_code='002015.SZ', start_date='20250101', end_date='20261231')

#基于fields参数指定输出字段
df = pro.stk_shock(trade_date='20260416', fields='ts_code,trade_date,name,reason')
print(df)