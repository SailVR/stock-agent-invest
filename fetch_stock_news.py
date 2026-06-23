import akshare as ak
import pandas as pd
import time

def fetch_stock_news(stock_code: str, delay: float = 0.6):
    try:
        df = ak.stock_news_em(symbol=stock_code)
        print(f"成功获取新闻，共 {len(df)} 条")
        time.sleep(delay)
        return df
    except Exception as err:
        print(f"抓取异常：{err}")
        return pd.DataFrame()


if __name__ == "__main__":
    target_stock = "002202"
    news_df = fetch_stock_news(target_stock)

    if news_df.empty:
        print("未获取到任何新闻数据！")
    else:
        # 修正字段：文章来源 替代 新闻来源
        show_cols = ["发布时间", "新闻标题", "新闻内容", "文章来源", "新闻链接"]
        print("\n========== 新闻预览 ==========")
        print(news_df[show_cols].head(10))

        save_name = f"个股新闻_{target_stock}.csv"
        news_df.to_csv(save_name, index=False, encoding="utf-8-sig")
        print(f"\n采集完成，共 {len(news_df)} 条新闻，保存至 {save_name}")