import pyarrow as pa

messages_schema = pa.struct(
    [
        pa.field("role", pa.string()),
        pa.field("content", pa.list_(pa.string())),
        pa.field("loss_mask", pa.list_(pa.float64())),
        pa.field("name", pa.string()),
    ]
)

schema = pa.schema(
    [
        pa.field("messages", pa.list_(messages_schema)),
        pa.field("domain", pa.string()),
        pa.field("cat", pa.string()),
        pa.field("source", pa.string()),
        pa.field("distill", pa.string()),
        pa.field("budget_tokens", pa.int64()),
    ]
)
