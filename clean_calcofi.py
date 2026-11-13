"""
Load and clean the CalCOFI bottle data, compute TEOS-10 density with gsw.

Inputs (from the CalCOFI Bottle Database CSV download):
    bottle.csv  and  cast.csv  
Output:
    calcofi_clean.csv  with columns Year, Lat, Lon, p, SA, CT, rho

"""
import numpy as np
import pandas as pd
import gsw

BOTTLE = "bottle.csv"
CAST = "cast.csv"
YEARS = (None, None)          # e.g. (1984, 2019); None = no limit. Decide after the diagnostics.

# load 
btl = pd.read_csv(BOTTLE, encoding="latin-1", low_memory=False)
cast = pd.read_csv(CAST, encoding="latin-1", low_memory=False)


print("bottle columns:", list(btl.columns)[:20], "...")
print("cast columns:  ", list(cast.columns)[:20], "...")
COL = dict(key="Cst_Cnt", depth="Depthm", T="T_degC", S="Salnty",
           Tq="T_qual", Sq="S_qual", Pq="P_qual",
           year="Year", lat="Lat_Dec", lon="Lon_Dec")

btl = btl[[COL[k] for k in ("key", "depth", "T", "S", "Tq", "Sq", "Pq")]]
cast = cast[[COL[k] for k in ("key", "year", "lat", "lon")]]
df = btl.merge(cast, on=COL["key"], how="left")
n0 = len(df)
print(f"rows after merge: {n0}")

# flags
# Look at the flag values before deciding what to keep
for k in ("Tq", "Sq", "Pq"):
    print(f"\n{COL[k]} value counts:\n", df[COL[k]].value_counts(dropna=False).head(10))

# an empty flag means "no problem reported", in the 
# CalCOFI code definitions 6 is also a good sample, just means 'Data taken from the CTD sensor'. 
# There are other flags which could be considered, cf teh CalCOFI code definitions.
EXTRA_OK = [6]
for k in ("Tq", "Sq"):
    df = df[df[COL[k]].isna() | df[COL[k]].isin(EXTRA_OK)]
    
# clean
df = df.rename(columns={COL["depth"]: "depth", COL["T"]: "temp", COL["S"]: "SP",
                        COL["year"]: "Year", COL["lat"]: "Lat", COL["lon"]: "Lon"})
df = df.dropna(subset=["depth", "temp", "SP", "Lat", "Lon", "Year"])

# Physical range filters, temp from -2 to 35 C, practical salinity from 30 to 40, depth >= 0
df = df[(df.depth >= 0) & df.temp.between(-2, 35) & df.SP.between(30, 40)]
df = df.drop_duplicates(subset=["Lat", "Lon", "Year", "depth", "temp", "SP"])

if YEARS[0] is not None:
    df = df[df.Year >= YEARS[0]]
if YEARS[1] is not None:
    df = df[df.Year <= YEARS[1]]
print(f"\nrows kept: {len(df)} of {n0}")

# TEOS-10
lat, lon = df.Lat.values, df.Lon.values
p = gsw.p_from_z(-df.depth.values, lat)        # dbar
SA = gsw.SA_from_SP(df.SP.values, p, lon, lat)  # g/kg
CT = gsw.CT_from_t(SA, df["temp"].values, p)       # deg C
rho = gsw.rho(SA, CT, p)                        # kg/m^3, ground truth

out = pd.DataFrame({"Year": df.Year.values, "Lat": lat, "Lon": lon,
                    "p": p, "SA": SA, "CT": CT, "rho": rho})
out = out.replace([np.inf, -np.inf], np.nan).dropna()
print(f"rows after gsw: {len(out)}")

# diagnostics
print("\nSamples per pressure bin (dbar):")
print(pd.cut(out.p, [0, 100, 200, 500, 1000, 2000, 6000], right=False)
      .value_counts().sort_index())

out["decade"] = (out.Year // 10 * 10).astype(int)
print("\nSamples per decade (n_deep = p > 500 dbar):")
print(out.groupby("decade").agg(n=("p", "size"),
                                n_deep=("p", lambda s: int((s > 500).sum()))))
print("\nRanges:\n", out[["p", "SA", "CT", "rho"]].describe().loc[["min", "max"]])

out.drop(columns="decade").to_csv("calcofi_clean.csv", index=False)
print("\nsaved calcofi_clean.csv")
