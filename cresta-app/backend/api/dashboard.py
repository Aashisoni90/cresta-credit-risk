"""Dashboard API endpoints."""
from fastapi import APIRouter
from services.company_service import get_dashboard_summary

router = APIRouter()


@router.get("/dashboard/summary")
async def dashboard_summary():
    return get_dashboard_summary()
