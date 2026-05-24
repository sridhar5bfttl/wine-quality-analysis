import os
import numpy as np
import pandas as pd

def generate_oil_gas_dataset(output_path, seed=42):
    np.random.seed(seed)
    
    # 3 years of daily production data
    date_range = pd.date_range(start="2023-01-01", end="2025-12-31", freq="D")
    n_days = len(date_range)
    
    # 1. Base reservoir decline (Arps decline curve)
    # q(t) = q0 / (1 + b * d * t)**(1/b) -> simplified exponential for simplicity: q0 * e^(-d*t)
    t = np.arange(n_days)
    q0 = 5000.0  # Initial flow capacity (bbl/day)
    decline_rate = 0.0008  # Daily decline rate
    base_potential = q0 * np.exp(-decline_rate * t)
    
    # 2. Simulate Operator Controls: Choke Size Percentage
    # Operators adjust choke sizes. Let's model it as a random walk with boundaries (20% to 100%)
    choke_size = np.zeros(n_days)
    current_choke = 80.0
    for idx in range(n_days):
        # Step changes or random drift
        drift = np.random.normal(0, 1.5)
        current_choke = np.clip(current_choke + drift, 20.0, 100.0)
        # Occasional manual adjustments
        if idx % 120 == 0 and idx > 0:
            current_choke = np.random.choice([40.0, 60.0, 80.0, 100.0])
        choke_size[idx] = current_choke
        
    # 3. Operational Uptime (on_stream_hours)
    # Usually 24 hours, but with shutdowns
    on_stream_hours = np.full(n_days, 24.0)
    for idx in range(n_days):
        # Regular seasonal maintenance: 7 days in August (month 8)
        if date_range[idx].month == 8 and 10 <= date_range[idx].day <= 16:
            on_stream_hours[idx] = 0.0
        # Random equipment trips (1-2 days of reduced hours or total shutdown)
        elif np.random.rand() < 0.02:
            on_stream_hours[idx] = np.random.choice([0.0, 4.0, 12.0])
            
    # 4. Pressures and Temperatures (Physically correlated to choke and decline)
    # Bottomhole Pressure (BHP) decreases as reservoir declines and decreases with larger choke (drawdown)
    base_bhp = 3500.0 * np.exp(-0.0004 * t)  # Declines slowly
    drawdown = 8.0 * choke_size  # More flow = lower bottomhole pressure
    bottomhole_pressure = np.clip(base_bhp - drawdown + np.random.normal(0, 15, n_days), 500.0, 4500.0)
    
    # Wellhead Pressure (WHP) is proportional to BHP and choked flow
    wellhead_pressure = bottomhole_pressure * (1.0 - (choke_size / 150.0)) + np.random.normal(0, 10, n_days)
    wellhead_pressure = np.clip(wellhead_pressure, 100.0, 3000.0)
    
    # Bottomhole Temperature (BHT) is relatively constant deep in the reservoir
    bottomhole_temperature = 180.0 + np.random.normal(0, 0.5, n_days)
    
    # Wellhead Temperature (WHT) cools down if flow rate is low (less thermal energy reaching surface)
    # WHT rises with higher choke size and uptime
    base_wht = 60.0  # Ambient sea temp
    wht_heating = 70.0 * (choke_size / 100.0) * (on_stream_hours / 24.0)
    wellhead_temperature = base_wht + wht_heating + np.random.normal(0, 1.5, n_days)
    
    # 5. Volumetric Calculations
    # Oil production depends on base potential, choke size, bottomhole pressure drawdown, and uptime
    oil_volume = base_potential * (choke_size / 100.0) * (on_stream_hours / 24.0)
    oil_volume += np.random.normal(0, 0.02 * (oil_volume + 1))
    oil_volume = np.clip(oil_volume, 0.0, None)
    
    # Gas Volume: Associated gas defined by Gas-Oil Ratio (GOR)
    # Let's say GOR is around 1.3 MSCF/bbl, increasing slightly as pressure drops (gas breakout)
    gor = 1.2 + (0.3 * (1.0 - (bottomhole_pressure / 3500.0)))
    gas_volume = oil_volume * gor + np.random.normal(0, 0.05 * (oil_volume * gor + 1))
    gas_volume = np.clip(gas_volume, 0.0, None)
    
    # Water Volume: Water Cut starts low (1%) and increases sigmooidally as aquifer water encroaches
    # Water cut = 1 / (1 + e^(-(t - mid_point)/scale))
    mid_point = n_days * 0.6
    scale = n_days * 0.15
    water_cut = 0.01 + 0.85 / (1.0 + np.exp(-(t - mid_point) / scale))
    
    # Adjust total fluid production
    total_fluid = oil_volume / (1.0 - water_cut + 1e-6)
    water_volume = total_fluid * water_cut
    # If shut down, volumes are zero
    water_volume[on_stream_hours == 0.0] = 0.0
    oil_volume[on_stream_hours == 0.0] = 0.0
    gas_volume[on_stream_hours == 0.0] = 0.0
    
    # Create DataFrame
    df = pd.DataFrame({
        "date": date_range,
        "on_stream_hours": np.round(on_stream_hours, 1),
        "choke_size_percentage": np.round(choke_size, 2),
        "bottomhole_pressure_psi": np.round(bottomhole_pressure, 1),
        "bottomhole_temperature_f": np.round(bottomhole_temperature, 1),
        "wellhead_pressure_psi": np.round(wellhead_pressure, 1),
        "wellhead_temperature_f": np.round(wellhead_temperature, 1),
        "oil_volume_bbl": np.round(oil_volume, 2),
        "gas_volume_mscf": np.round(gas_volume, 2),
        "water_volume_bbl": np.round(water_volume, 2)
    })
    
    # Add calculated columns for diagnostics
    df["water_cut_percentage"] = np.round((df["water_volume_bbl"] / (df["oil_volume_bbl"] + df["water_volume_bbl"] + 1e-6)) * 100, 2)
    df["gor_mscf_bbl"] = np.round(df["gas_volume_mscf"] / (df["oil_volume_bbl"] + 1e-6), 4)
    
    # Zero out diagnostics if shutdown
    df.loc[df["on_stream_hours"] == 0, "water_cut_percentage"] = 0.0
    df.loc[df["on_stream_hours"] == 0, "gor_mscf_bbl"] = 0.0
    
    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Successfully generated dataset at {output_path} with {len(df)} rows.")

if __name__ == "__main__":
    generate_oil_gas_dataset("oil_gas_production/data/well_production_data.csv")
