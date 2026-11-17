"""Companies API endpoints."""
from fastapi import APIRouter, HTTPException, Query
from services.company_service import (
    get_company_list, get_company_detail, get_company_history,
    get_company_features, get_sectors
)

router = APIRouter()


@router.get("/companies")
async def list_companies(
    search: str = Query(None),
    sector: str = Query(None),
    page: int = Query(1, ge=1),
    per_page: int = Query(25, ge=1, le=100),
):
    return get_company_list(search=search, sector=sector, page=page, per_page=per_page)


@router.get("/companies/sectors")
async def list_sectors():
    return {"sectors": get_sectors()}


@router.get("/companies/{ticker}")
async def company_detail(ticker: str):
    result = get_company_detail(ticker)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Company {ticker} not found")
    return result


@router.get("/companies/{ticker}/history")
async def company_history(ticker: str):
    result = get_company_history(ticker)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Company {ticker} not found")
    return result


@router.get("/companies/{ticker}/features/{year}")
async def company_features(ticker: str, year: int):
    result = get_company_features(ticker, year)
    if result is None:
        raise HTTPException(status_code=404, detail=f"No features for {ticker} in {year}")
    return result
