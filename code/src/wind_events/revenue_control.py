"""Complete-revenue accounting and forecast-only co-located storage planning.

Browell (2018), Risk Constrained Trading Strategies for Stochastic Generation
with a Single-Price Balancing Market, doi:10.3390/en11061345, trading revenue
formulation: include forward sales plus signed imbalance receipts. Under a
fixed common contract, revenue differences isolate changes in delivery.

The receding-horizon linear programme is study-defined: wind-only charging,
export cap, curtailment, asymmetric efficiency and closed terminal inventory.
It maximizes forecast settlement receipts net of AC-throughput wear costs.
Only a passive metering/feasibility layer observes realized interval power.
"""
import numpy as np
from scipy.optimize import linprog
from .economics import Battery, imbalance_cashflow


def complete_revenue(actual, schedule, forward_price, buy_price, sell_price, dt=.5):
    actual,schedule,forward_price=np.broadcast_arrays(actual,schedule,forward_price)
    if not np.isfinite(forward_price).all():raise ValueError('Finite trade/reference prices required')
    settlement=imbalance_cashflow(actual,schedule,buy_price,sell_price,dt)
    return schedule*forward_price*dt-settlement['net_imbalance_cost_gbp']


def plan_storage(wind, price, inventory, battery, degradation=10., dt=.5, export_cap=48.3):
    wind=np.maximum(np.asarray(wind,float),0);price=np.asarray(price,float)
    n=len(wind)
    if n==0 or price.shape!=wind.shape or not np.isfinite([*wind,*price,inventory]).all():
        raise ValueError('Aligned finite forecast trajectories required')
    if degradation<0 or dt<=0 or export_cap<=0:raise ValueError('Invalid operating conditions')
    low=battery.minimum_fraction*battery.energy_mwh; high=battery.maximum_fraction*battery.energy_mwh
    if not low-1e-7<=inventory<=high+1e-7:raise ValueError('Initial state outside bounds')
    # Variables are charge, discharge, wind curtailment, and inventory at all nodes.
    cost=np.zeros(4*n+1);cost[:n]=dt*(price+degradation+1e-6)
    cost[n:2*n]=dt*(-price+degradation+1e-6);cost[2*n:3*n]=dt*price
    equal=np.zeros((n,4*n+1))
    for i in range(n):
        equal[i,i]=-battery.charge_efficiency*dt;equal[i,n+i]=dt/battery.discharge_efficiency
        equal[i,3*n+i]=-1;equal[i,3*n+i+1]=1
    upper=np.zeros((2*n,4*n+1));rhs=np.r_[wind,export_cap-wind]
    for i in range(n):
        upper[i,i]=1;upper[i,2*n+i]=1
        upper[n+i,i]=-1;upper[n+i,n+i]=1;upper[n+i,2*n+i]=-1
    bounds=[(0,battery.power_mw)]*(2*n)+[(0,float(w)) for w in wind]+[(low,high)]*(n+1)
    bounds[3*n]=(inventory,inventory);bounds[-1]=(low,low)
    result=linprog(cost,A_ub=upper,b_ub=rhs,A_eq=equal,b_eq=np.zeros(n),bounds=bounds,method='highs')
    if not result.success:raise RuntimeError(result.message)
    c,d,s=np.split(result.x[:3*n],3)
    if np.minimum(c,d).max()>1e-6:raise AssertionError('Simultaneous cycling in optimizer')
    return {'charge':c,'discharge':d,'curtail':s,'inventory':result.x[3*n:]}


def deliver_storage(wind, charge_request, discharge_request, curtail_request, inventory,
                    battery, remaining_intervals, dt=.5, export_cap=48.3):
    """Passive average-interval tracking, not future information for the planner.

    Realized wind clips scheduled charging; excess net export curtails wind.
    The reachable terminal band guarantees closure even after forecast errors.
    Native sub-interval ramps are outside this half-hour engineering model.
    """
    if not np.isfinite([wind,charge_request,discharge_request,curtail_request,inventory]).all():
        raise ValueError('Finite meter, requests and inventory required')
    if min(charge_request,discharge_request)<-1e-6 or min(charge_request,discharge_request)>1e-6:
        raise ValueError('A one-direction battery request is required')
    low=battery.minimum_fraction*battery.energy_mwh;high=battery.maximum_fraction*battery.energy_mwh
    if remaining_intervals<0 or not low-1e-6<=inventory<=high+1e-6:
        raise ValueError('Invalid state or remaining horizon')
    allowed_high=min(high,low+remaining_intervals*battery.power_mw*dt/battery.discharge_efficiency)
    mandatory_discharge=max(0,(inventory-allowed_high)*battery.discharge_efficiency/dt)
    if mandatory_discharge>1e-8:
        charge=0.;discharge=mandatory_discharge
    else:
        discharge=min(max(discharge_request,0),battery.power_mw,max(inventory-low,0)*battery.discharge_efficiency/dt)
        charge=min(max(charge_request,0),battery.power_mw,max(wind,0),
                   max(allowed_high-inventory,0)/(battery.charge_efficiency*dt))
    state=inventory+charge*battery.charge_efficiency*dt-discharge*dt/battery.discharge_efficiency
    spill=min(max(curtail_request,0),max(wind-charge,0))
    spill=max(spill,wind-charge+discharge-export_cap)
    delivered=wind-charge+discharge-spill
    return {'delivered':delivered,'charge':charge,'discharge':discharge,'curtail':spill,
            'inventory':state,'throughput':(charge+discharge)*dt,
            'terminal_safety_override':mandatory_discharge>1e-8}
