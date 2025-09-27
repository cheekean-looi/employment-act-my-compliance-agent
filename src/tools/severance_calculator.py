#!/usr/bin/env python3
"""
Severance Calculator Tool for Employment Act Malaysia
Pure arithmetic calculator with transparent formulas and disclaimers.
Users must verify against current law.
"""

from typing import Dict, Optional, Union, Tuple, List
from dataclasses import dataclass
from enum import Enum

class TerminationReason(Enum):
    """Valid termination reasons under Employment Act."""
    RETRENCHMENT = "retrenchment"
    CONTRACT_EXPIRY = "contract_expiry"
    RESIGNATION = "resignation"
    MUTUAL_CONSENT = "mutual_consent"
    MISCONDUCT = "misconduct"
    RETIREMENT = "retirement"

@dataclass
class SeveranceResult:
    """Result from severance calculation."""
    severance_pay: float
    notice_pay: float
    annual_leave_pay: float
    total_compensation: float
    calculation_breakdown: Dict[str, any]
    disclaimers: List[str]
    formula_used: str

class SeveranceCalculator:
    """
    Safe severance pay calculator implementing Employment Act Malaysia formulas.
    Pure arithmetic with transparent calculations and mandatory disclaimers.
    """
    
    def __init__(self):
        """Initialize calculator with Employment Act formulas."""
        
        # Notice period requirements (in weeks) based on service length
        self.notice_periods = {
            "under_2_years": 4,
            "2_to_5_years": 6, 
            "over_5_years": 8
        }
        
        # Severance pay formulas
        self.severance_formulas = {
            "retrenchment": {
                "under_2_years": lambda years, monthly_wage: years * 10 * (monthly_wage / 30),
                "2_to_5_years": lambda years, monthly_wage: years * 15 * (monthly_wage / 30),
                "over_5_years": lambda years, monthly_wage: years * 20 * (monthly_wage / 30)
            }
        }
        
        # Standard disclaimers
        self.disclaimers = [
            "This calculation is based on general Employment Act provisions and may not reflect your specific situation.",
            "Actual entitlements may vary based on employment contract terms, company policies, or special circumstances.",
            "Please verify calculations against current Employment Act provisions and consult HR or legal counsel.",
            "This tool does not constitute legal advice and should not be used as the sole basis for any decisions.",
            "Calculations assume standard employment under Employment Act - different rules may apply for specific industries."
        ]
    
    def calculate_severance(self, 
                          monthly_wage: float,
                          years_of_service: float,
                          termination_reason: Union[str, TerminationReason],
                          include_notice: bool = True,
                          annual_leave_days: Optional[int] = None) -> SeveranceResult:
        """
        Calculate severance compensation based on Employment Act formulas.
        
        Args:
            monthly_wage: Monthly basic wage (RM)
            years_of_service: Years of continuous service
            termination_reason: Reason for termination
            include_notice: Whether to include notice pay
            annual_leave_days: Outstanding annual leave days
            
        Returns:
            SeveranceResult with detailed breakdown
        """
        
        # Input validation
        if monthly_wage <= 0:
            raise ValueError("Monthly wage must be positive")
        if years_of_service < 0:
            raise ValueError("Years of service cannot be negative")
        
        # Convert string to enum if needed
        if isinstance(termination_reason, str):
            try:
                termination_reason = TerminationReason(termination_reason.lower())
            except ValueError:
                raise ValueError(f"Invalid termination reason: {termination_reason}")
        
        # Initialize result components
        severance_pay = 0.0
        notice_pay = 0.0
        annual_leave_pay = 0.0
        breakdown = {}
        formula_used = ""
        
        # Calculate severance pay based on reason
        if termination_reason == TerminationReason.RETRENCHMENT:
            severance_pay, breakdown, formula_used = self._calculate_retrenchment_pay(
                monthly_wage, years_of_service
            )
        elif termination_reason in [TerminationReason.CONTRACT_EXPIRY, TerminationReason.MUTUAL_CONSENT]:
            # May be entitled to severance based on contract
            severance_pay = 0.0
            formula_used = "Contract expiry/mutual consent - refer to employment contract"
            breakdown = {"note": "Severance depends on employment contract terms"}
        elif termination_reason == TerminationReason.RESIGNATION:
            severance_pay = 0.0
            formula_used = "Resignation - no severance pay"
            breakdown = {"note": "Employee resignation typically has no severance entitlement"}
        elif termination_reason == TerminationReason.MISCONDUCT:
            severance_pay = 0.0
            notice_pay = 0.0  # No notice for misconduct dismissal
            formula_used = "Misconduct dismissal - no severance/notice pay"
            breakdown = {"note": "Dismissal for misconduct typically has no compensation"}
        else:
            formula_used = "Standard termination - refer to Employment Act"
        
        # Calculate notice pay if applicable
        if include_notice and termination_reason != TerminationReason.MISCONDUCT:
            notice_pay, notice_breakdown = self._calculate_notice_pay(monthly_wage, years_of_service)
            breakdown["notice_calculation"] = notice_breakdown
        
        # Calculate annual leave pay if provided
        if annual_leave_days and annual_leave_days > 0:
            annual_leave_pay = self._calculate_annual_leave_pay(monthly_wage, annual_leave_days)
            breakdown["annual_leave_calculation"] = {
                "outstanding_days": annual_leave_days,
                "daily_rate": monthly_wage / 30,
                "calculation": f"{annual_leave_days} days × RM{monthly_wage/30:.2f}/day"
            }
        
        # Total compensation
        total_compensation = severance_pay + notice_pay + annual_leave_pay
        
        return SeveranceResult(
            severance_pay=round(severance_pay, 2),
            notice_pay=round(notice_pay, 2),
            annual_leave_pay=round(annual_leave_pay, 2),
            total_compensation=round(total_compensation, 2),
            calculation_breakdown=breakdown,
            disclaimers=self.disclaimers,
            formula_used=formula_used
        )
    
    def _calculate_retrenchment_pay(self, monthly_wage: float, years_of_service: float) -> Tuple[float, Dict, str]:
        """Calculate retrenchment/redundancy pay based on Employment Act."""
        
        if years_of_service < 1:
            return 0.0, {"note": "Less than 1 year service - no retrenchment benefit"}, "Under 1 year service"
        
        # Determine rate based on service length
        if years_of_service < 2:
            days_per_year = 10
            category = "under_2_years"
        elif years_of_service <= 5:
            days_per_year = 15
            category = "2_to_5_years"
        else:
            days_per_year = 20
            category = "over_5_years"
        
        # Calculate: years of service × days per year × daily wage
        daily_wage = monthly_wage / 30
        severance_pay = years_of_service * days_per_year * daily_wage
        
        formula = f"{years_of_service} years × {days_per_year} days/year × RM{daily_wage:.2f}/day"
        
        breakdown = {
            "service_years": years_of_service,
            "days_per_year": days_per_year,
            "daily_wage": daily_wage,
            "formula": formula,
            "category": category
        }
        
        return severance_pay, breakdown, f"Retrenchment pay ({category}): {formula}"
    
    def _calculate_notice_pay(self, monthly_wage: float, years_of_service: float) -> Tuple[float, Dict]:
        """Calculate notice pay based on Employment Act requirements."""
        
        # Determine notice period
        if years_of_service < 2:
            notice_weeks = self.notice_periods["under_2_years"]
        elif years_of_service <= 5:
            notice_weeks = self.notice_periods["2_to_5_years"]
        else:
            notice_weeks = self.notice_periods["over_5_years"]
        
        # Calculate weekly wage and notice pay
        weekly_wage = monthly_wage / 4.33  # Average weeks per month
        notice_pay = notice_weeks * weekly_wage
        
        breakdown = {
            "notice_weeks": notice_weeks,
            "weekly_wage": weekly_wage,
            "formula": f"{notice_weeks} weeks × RM{weekly_wage:.2f}/week"
        }
        
        return notice_pay, breakdown
    
    def _calculate_annual_leave_pay(self, monthly_wage: float, days: int) -> float:
        """Calculate outstanding annual leave payment."""
        daily_wage = monthly_wage / 30
        return days * daily_wage
    
    def get_calculation_summary(self, result: SeveranceResult) -> str:
        """Generate human-readable calculation summary."""
        summary = [
            "SEVERANCE CALCULATION SUMMARY",
            "=" * 35,
            f"Severance Pay: RM {result.severance_pay:,.2f}",
            f"Notice Pay: RM {result.notice_pay:,.2f}",
            f"Annual Leave Pay: RM {result.annual_leave_pay:,.2f}",
            f"TOTAL COMPENSATION: RM {result.total_compensation:,.2f}",
            "",
            f"Formula Used: {result.formula_used}",
            "",
            "CALCULATION BREAKDOWN:",
        ]
        
        # Add breakdown details
        for key, value in result.calculation_breakdown.items():
            if isinstance(value, dict):
                summary.append(f"\n{key.replace('_', ' ').title()}:")
                for k, v in value.items():
                    summary.append(f"  {k}: {v}")
            else:
                summary.append(f"{key}: {value}")
        
        # Add disclaimers
        summary.extend([
            "",
            "IMPORTANT DISCLAIMERS:",
            "-" * 21
        ])
        for i, disclaimer in enumerate(result.disclaimers, 1):
            summary.append(f"{i}. {disclaimer}")
        
        return "\n".join(summary)

def test_calculator():
    """Test the severance calculator with various scenarios."""
    print("🧮 Testing Severance Calculator")
    print("=" * 40)
    
    calculator = SeveranceCalculator()
    
    # Test scenarios
    test_cases = [
        {
            "name": "Retrenchment - 3 years service",
            "monthly_wage": 3000,
            "years_of_service": 3,
            "termination_reason": "retrenchment",
            "annual_leave_days": 10
        },
        {
            "name": "Retrenchment - 7 years service",
            "monthly_wage": 5000,
            "years_of_service": 7,
            "termination_reason": "retrenchment",
            "annual_leave_days": 15
        },
        {
            "name": "Resignation",
            "monthly_wage": 2500,
            "years_of_service": 2,
            "termination_reason": "resignation",
            "include_notice": False
        },
        {
            "name": "Misconduct dismissal",
            "monthly_wage": 4000,
            "years_of_service": 5,
            "termination_reason": "misconduct"
        }
    ]
    
    for test_case in test_cases:
        case_name = test_case.pop('name')  # Remove name before passing to function
        print(f"\n--- {case_name} ---")
        try:
            result = calculator.calculate_severance(**test_case)
            print(f"Total compensation: RM {result.total_compensation:,.2f}")
            print(f"Formula: {result.formula_used}")
            
            if result.severance_pay > 0:
                print(f"  Severance: RM {result.severance_pay:,.2f}")
            if result.notice_pay > 0:
                print(f"  Notice: RM {result.notice_pay:,.2f}")
            if result.annual_leave_pay > 0:
                print(f"  Annual leave: RM {result.annual_leave_pay:,.2f}")
                
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_calculator()