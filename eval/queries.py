# Ground-truth pairs for evaluating RAG retrieval on a generic sales/e-commerce dataset.
# Each entry: (natural language query, list of columns that must appear in top-5 retrieved chunks)

QUERIES = [
    ("show distribution of sales", ["sales"]),
    ("plot sales by region", ["region", "sales"]),
    ("compare profit across categories", ["category", "profit"]),
    ("customer segment distribution", ["segment", "customer"]),
    ("monthly trend of orders", ["order date"]),
    ("ranked product name by total sales", ["product name", "sales"]),
    ("sales by segment", ["segment", "sales"]),
    ("average discount by category", ["discount", "category"]),
    ("sales trend over time", ["order date", "sales"]),
    ("heatmap of sales profit discount quantity", ["sales", "profit"]),
    ("bar chart of order count by country", ["country", "order"]),
    ("sales distribution by sub-category", ["sub-category", "sales"]),
    ("sales split by segment", ["segment", "sales"]),
    ("profit by sub-category", ["sub-category", "profit"]),
    ("box plot of sales by segment", ["sales", "segment"]),
    ("total sales per region", ["region", "sales"]),
    ("customer name ranked by total sales", ["customer name", "sales"]),
    ("sales by ship mode", ["ship mode", "sales"]),
]