import geopandas as gpd
import pickle
import pandas as pd

#%%
shpFileLoc = 'Data/west_midlands_OAs/west_midlands_OAs.shp'
oaInfoLoc = 'Data/oa_info.csv'
area = 'E08000025'

#%%

wm_oas = gpd.read_file(shpFileLoc)
wm_oas = wm_oas[wm_oas['LAD11CD'] == area]
oa_info = pd.read_csv(oaInfoLoc)
oa_info = oa_info.merge(wm_oas[['OA11CD']], left_on = 'oa_id', right_on = 'OA11CD', how = 'inner')
#OA List - list of OA IDs - used for subseuqent SQL query
oaList = tuple(list(set(list(oa_info['oa_id']))))

#%%

#Which OA are walkable from one another and in how long
f = open('Data/features/walkableOAs.txt', 'rb')
walkableOAs = pickle.load(f)
f.close()

#%%

oasReachableNMin = {}

for o in oaList:
    oasReachableNMin[o] = {
        2:[o],
        4:[o],
        6:[o],
        8:[o],
        10:[o],
        12:[o],
        14:[o],
        16:[o],
        18:[o],
        20:[o]
        }
    try:
        for k,v in walkableOAs[o].items():
            if k != o:
                if v[0] == 2:
                    oasReachableNMin[o][2].append(k)
                    oasReachableNMin[o][4].append(k)
                    oasReachableNMin[o][6].append(k)
                    oasReachableNMin[o][8].append(k)
                    oasReachableNMin[o][10].append(k)
                    oasReachableNMin[o][12].append(k)
                    oasReachableNMin[o][14].append(k)
                    oasReachableNMin[o][16].append(k)
                    oasReachableNMin[o][18].append(k)
                    oasReachableNMin[o][20].append(k)
                elif v[0] == 4:
                    oasReachableNMin[o][4].append(k)
                    oasReachableNMin[o][6].append(k)
                    oasReachableNMin[o][8].append(k)
                    oasReachableNMin[o][10].append(k)
                    oasReachableNMin[o][12].append(k)
                    oasReachableNMin[o][14].append(k)
                    oasReachableNMin[o][16].append(k)
                    oasReachableNMin[o][18].append(k)
                    oasReachableNMin[o][20].append(k)
                elif v[0] == 6:
                    oasReachableNMin[o][6].append(k)
                    oasReachableNMin[o][8].append(k)
                    oasReachableNMin[o][10].append(k)
                    oasReachableNMin[o][12].append(k)
                    oasReachableNMin[o][14].append(k)
                    oasReachableNMin[o][16].append(k)
                    oasReachableNMin[o][18].append(k)
                    oasReachableNMin[o][20].append(k)
                elif v[0] == 8:
                    oasReachableNMin[o][8].append(k)
                    oasReachableNMin[o][10].append(k)
                    oasReachableNMin[o][12].append(k)
                    oasReachableNMin[o][14].append(k)
                    oasReachableNMin[o][16].append(k)
                    oasReachableNMin[o][18].append(k)
                    oasReachableNMin[o][20].append(k)
                elif v[0] == 10:
                    oasReachableNMin[o][10].append(k)
                    oasReachableNMin[o][12].append(k)
                    oasReachableNMin[o][14].append(k)
                    oasReachableNMin[o][16].append(k)
                    oasReachableNMin[o][18].append(k)
                    oasReachableNMin[o][20].append(k)          
                elif v[0] == 12:
                    oasReachableNMin[o][12].append(k)
                    oasReachableNMin[o][14].append(k)
                    oasReachableNMin[o][16].append(k)
                    oasReachableNMin[o][18].append(k)
                    oasReachableNMin[o][20].append(k)       
                elif v[0] == 14:
                    oasReachableNMin[o][14].append(k)
                    oasReachableNMin[o][16].append(k)
                    oasReachableNMin[o][18].append(k)
                    oasReachableNMin[o][20].append(k)        
                elif v[0] == 16:
                    oasReachableNMin[o][16].append(k)
                    oasReachableNMin[o][18].append(k)
                    oasReachableNMin[o][20].append(k)       
                elif v[0] == 18:
        
                    oasReachableNMin[o][18].append(k)
                    oasReachableNMin[o][20].append(k)
                elif v[0] == 20:
        
                    oasReachableNMin[o][20].append(k)     
    except:
        print(o)

#%%

f = open('Data/features/OAsReachableOnFoot.txt', 'wb')
pickle.dump(oasReachableNMin,f)
f.close()