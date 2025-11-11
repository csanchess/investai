import yfinance as yf
import os

def build_financial_esg_corpus(tickers, num_days=180, output_file="data/financial_esg_corpus.txt"):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    corpus_lines = []

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=f"{num_days}d")
            info = stock.info
            esg = getattr(stock, "sustainability", None)

            corpus_lines.append(f"=== COMPANY: {ticker} ===")
            corpus_lines.append(f"Sector: {info.get('sector', 'N/A')}")
            corpus_lines.append(f"Industry: {info.get('industry', 'N/A')}")
            corpus_lines.append(f"Market Cap: {info.get('marketCap', 'N/A')}")
            corpus_lines.append(f"Country: {info.get('country', 'N/A')}")
            corpus_lines.append(f"Summary: {info.get('longBusinessSummary', '')}")
            corpus_lines.append("")

            if esg is not None:
                corpus_lines.append("ESG Scores:")
                for metric, val in esg.itertuples():
                    corpus_lines.append(f"{metric}: {val}")
                corpus_lines.append("")

            hist = hist.tail(10)
            for _, row in hist.iterrows():
                corpus_lines.append(
                    f"Date: {row.name.date()}, Open: {row['Open']:.2f}, Close: {row['Close']:.2f}, Volume: {row['Volume']}"
                )
            corpus_lines.append("\n\n")

        except Exception as e:
            print(f"⚠️ Error retrieving {ticker}: {e}")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(corpus_lines))

    print(f"✅ Dataset saved to {output_file}")
    return output_file

