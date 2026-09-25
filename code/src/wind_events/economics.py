"""Unit-explicit energy error accounting and physically feasible storage steps.

Elexon BSC Section T and the Electricity Trading Arrangements guide specify
settlement-period imbalance cashflows. A forecast here defines an experimental
schedule; actual contractual positions are required for a realized invoice.
"""
from dataclasses import dataclass
import numpy as np


def imbalance_cashflow(actual_mw, scheduled_mw, buy_gbp_per_mwh, sell_gbp_per_mwh, interval_hours=.5):
    actual, scheduled, buy, sell = np.broadcast_arrays(actual_mw, scheduled_mw, buy_gbp_per_mwh, sell_gbp_per_mwh)
    if interval_hours <= 0 or not all(np.isfinite(x).all() for x in (actual, scheduled, buy, sell)):
        raise ValueError("Finite aligned power/prices and positive interval required")
    deviation_mwh = (actual-scheduled)*interval_hours
    cost = np.maximum(-deviation_mwh, 0)*buy-np.maximum(deviation_mwh, 0)*sell
    return {"deviation_mwh": deviation_mwh, "net_imbalance_cost_gbp": cost,
            "shortfall_purchase_gbp": np.maximum(-deviation_mwh, 0)*buy,
            "surplus_sale_gbp": np.maximum(deviation_mwh, 0)*sell}


def excess_error_energy(actual_mw, scheduled_mw, tolerance_mw, interval_hours):
    """Mathematical tolerance experiment; a policy must supply its own basis."""
    if tolerance_mw < 0 or interval_hours <= 0:
        raise ValueError("Nonnegative tolerance and positive interval required")
    return np.maximum(np.abs(np.asarray(actual_mw)-np.asarray(scheduled_mw))-tolerance_mw, 0)*interval_hours


def settlement_opportunity_cost(actual_mw, scheduled_mw, contract_gbp_per_mwh,
                                buy_gbp_per_mwh, sell_gbp_per_mwh, interval_hours=.5):
    """Revenue difference from a perfect-energy schedule at a declared contract price.

    Include schedule revenue as well as imbalance cashflow. Without this term,
    a change in scheduled energy can be mistaken for a forecast-quality gain.
    The contract price is a required observed input, never inferred from nMAE.
    """
    actual, scheduled, contract = np.broadcast_arrays(actual_mw, scheduled_mw, contract_gbp_per_mwh)
    if not np.isfinite(contract).all():
        raise ValueError("Finite observed contract/reference price required")
    imbalance = imbalance_cashflow(actual, scheduled, buy_gbp_per_mwh, sell_gbp_per_mwh, interval_hours)
    return imbalance["net_imbalance_cost_gbp"]+(actual-scheduled)*interval_hours*contract


def rte_imbalance_settlement_price(pmp_h, pmp_b, vwap_sign, trend, imbalance_sign, k_h, k_b):
    """Apply the RTE balance-responsible-party ISP rule to one interval.

    RTE Market Rules, Balance Responsible Party chapter (ISP): for positive or
    nil VWAP, a positive imbalance uses PMP*(1-k) and a negative imbalance
    uses PMP*(1+k); for negative VWAP the signs reverse. ``trend`` is ``up``
    or ``down`` for the French system balancing trend, and ``imbalance_sign``
    is ``positive`` or ``negative`` for the BRP perimeter. The required PMP,
    VWAP sign and published k factors remain explicit observed inputs.
    """
    if vwap_sign not in ("positive", "negative") or trend not in ("up", "down") or imbalance_sign not in ("positive", "negative"):
        raise ValueError("Invalid RTE sign category")
    if not np.isfinite([pmp_h, pmp_b, k_h, k_b]).all():
        raise ValueError("Finite RTE price and coefficient inputs required")
    base = pmp_h if trend == "up" else pmp_b
    k = k_h if trend == "up" else k_b
    plus_factor = (vwap_sign == "positive") == (imbalance_sign == "negative")
    return float(base*(1+k if plus_factor else 1-k))


@dataclass(frozen=True)
class Battery:
    power_mw: float
    energy_mwh: float
    charge_efficiency: float = .92
    discharge_efficiency: float = .92
    minimum_fraction: float = .1
    maximum_fraction: float = .9

    def __post_init__(self):
        if self.power_mw < 0 or self.energy_mwh < 0 or not 0 < self.charge_efficiency <= 1 or not 0 < self.discharge_efficiency <= 1:
            raise ValueError("Invalid battery power, energy or efficiency")
        if not 0 <= self.minimum_fraction <= self.maximum_fraction <= 1:
            raise ValueError("Invalid SOC fraction bounds")


def dispatch_step(actual_mw, scheduled_mw, stored_mwh, battery, interval_hours=.5):
    """Ex-post interval correction; actual interval power is observed information.

    Use as a correction-capability benchmark, not as an advance dispatch rule.
    Real-time deployment requires within-interval metering and control data.
    """
    if interval_hours <= 0:
        raise ValueError("Positive interval required")
    low, high = battery.minimum_fraction*battery.energy_mwh, battery.maximum_fraction*battery.energy_mwh
    if not low-1e-10 <= stored_mwh <= high+1e-10:
        raise ValueError("Initial inventory violates storage bounds")
    surplus = actual_mw-scheduled_mw
    charge = min(max(surplus, 0), battery.power_mw, max(high-stored_mwh, 0)/(battery.charge_efficiency*interval_hours))
    discharge = min(max(-surplus, 0), battery.power_mw, max(stored_mwh-low, 0)*battery.discharge_efficiency/interval_hours)
    new_inventory = stored_mwh+battery.charge_efficiency*charge*interval_hours-discharge*interval_hours/battery.discharge_efficiency
    return {"metered_mw": actual_mw-charge+discharge, "stored_mwh": new_inventory, "charge_mw": charge, "discharge_mw": discharge,
            "throughput_mwh": (charge+discharge)*interval_hours}


def planned_inventory_step(predicted_mw, commitment_mw, stored_mwh, battery,
                           remaining_intervals, terminal_mwh, interval_hours=.5):
    """Plan an action from issued predictions, inventory and known constraints.

    The reachable terminal-inventory band enforces daily energy neutrality
    without looking at future generation or prices. Positive export is discharge.
    Grid charging is allowed in this engineering configuration.
    """
    if remaining_intervals < 0 or interval_hours <= 0:
        raise ValueError("Nonnegative remaining horizon and positive time step required")
    if not np.isfinite([predicted_mw, commitment_mw, stored_mwh, terminal_mwh]).all():
        raise ValueError("Finite predictions and state required")
    lower = battery.minimum_fraction*battery.energy_mwh
    upper = battery.maximum_fraction*battery.energy_mwh
    if not lower-1e-8 <= stored_mwh <= upper+1e-8 or not lower-1e-8 <= terminal_mwh <= upper+1e-8:
        raise ValueError("Current and terminal inventory must satisfy SOC bounds")
    charge_step = battery.power_mw*battery.charge_efficiency*interval_hours
    discharge_step = battery.power_mw*interval_hours/battery.discharge_efficiency
    lo = max(lower, stored_mwh-discharge_step, terminal_mwh-remaining_intervals*charge_step)
    hi = min(upper, stored_mwh+charge_step, terminal_mwh+remaining_intervals*discharge_step)
    if lo > hi+1e-8:
        raise ValueError("Terminal inventory is unreachable")
    wanted = commitment_mw-predicted_mw
    desired = stored_mwh-wanted*interval_hours/(battery.discharge_efficiency if wanted >= 0 else 1/battery.charge_efficiency)
    new = float(np.clip(desired, lo, hi))
    charge = max(new-stored_mwh, 0)/(battery.charge_efficiency*interval_hours)
    discharge = max(stored_mwh-new, 0)*battery.discharge_efficiency/interval_hours
    return {"stored_mwh": new, "charge_mw": charge, "discharge_mw": discharge,
            "battery_export_mw": discharge-charge, "throughput_mwh": (charge+discharge)*interval_hours}
