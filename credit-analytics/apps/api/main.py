from fastapi import FastAPI

app = FastAPI(
    title="Credit Analytics API",
    description="API for accessing credit trading signals and backtest results.",
    version="0.1.0",
)

@app.get("/")
async def root():
    return {"message": "Credit Analytics API"}

