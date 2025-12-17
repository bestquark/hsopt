from pymol import cmd,stored

set depth_cue, 1
set fog_start, 0.4

set_color b_col, [36,36,85]
set_color t_col, [10,10,10]
set bg_rgb_bottom, b_col
set bg_rgb_top, t_col      
set bg_gradient

set  spec_power  =  200
set  spec_refl   =  0

load "data/3GPH_chainA_fixed.pdb", protein
create ligands, protein and organic
select xlig, protein and organic
delete xlig

hide everything, all

color white, elem c
color bluewhite, protein
#show_as cartoon, protein
show surface, protein
#set transparency, 0.15

show sticks, ligands
set stick_color, magenta




# SAS points

load "data/3GPH_chainA_fixed.pdb_points.pdb.gz", points
hide nonbonded, points
show nb_spheres, points
set sphere_scale, 0.2, points
cmd.spectrum("b", "green_red", selection="points", minimum=0, maximum=0.7)


stored.list=[]
cmd.iterate("(resn STP)","stored.list.append(resi)")    # read info about residues STP
lastSTP=stored.list[-1] # get the index of the last residue
hide lines, resn STP

cmd.select("rest", "resn STP and resi 0")

for my_index in range(1,int(lastSTP)+1): cmd.select("pocket"+str(my_index), "resn STP and resi "+str(my_index))
for my_index in range(1,int(lastSTP)+1): cmd.show("spheres","pocket"+str(my_index))
for my_index in range(1,int(lastSTP)+1): cmd.set("sphere_scale","0.4","pocket"+str(my_index))
for my_index in range(1,int(lastSTP)+1): cmd.set("sphere_transparency","0.1","pocket"+str(my_index))



set_color pcol1 = [0.361,0.576,0.902]
select surf_pocket1, protein and id [4362,4378,4361,1228,4366,2806,2931,7289,7291,1379,1372,5509,5513,6639,6634,6617,6630,6593,6628,5424,5449,5451,5455,5494,6504,5910,6516,6518,6519,6540,6543,2870,2873,2874,2878,2882,4358,4364,3462,1277,1366,1353,1356,6645,1360,5504,1143,1145,6594,5547,1369,1350,1479,1484,6603,2861,2856,3415,3466,4426,4422,4432,4434,5439,4428,4440,6528,6637,6703,6524,6532,6697,6699,6774,4479,5430,5433,6505,5336,5339,5343,4409,4374,4376,4382,4384,2811,4377,2807,6651,6693,6761,6649,2399,4328,1684,1680,2803] 
set surface_color,  pcol1, surf_pocket1 
set_color pcol2 = [0.278,0.341,0.702]
select surf_pocket2, protein and id [1204,3084,3083,3087,3089,3094,1164,1165,5522,5831,729,732,3134,3137,3097,379,3102,3103,3026,286,777,446,5863,448,776,5867,3118,697,1177,1186,2935,7283,1210,5518,5525,5528,5489,7272,2925,375,3034,3035,5478,5479,5482,1170,1226,7275,7279,3067,3040] 
set surface_color,  pcol2, surf_pocket2 
set_color pcol3 = [0.424,0.361,0.902]
select surf_pocket3, protein and id [5418,5985,5994,5924,5927,5929,5986,5990,5923,5982,5922,5930,5932,5935,5981,7205,7208,7262,361,399,403,458,456,460,360,386,387,388,397,844,530,531,5376,5387,5385,5389,6120,6117,6121,5393,4546,7192,7172,7188] 
set surface_color,  pcol3, surf_pocket3 
set_color pcol4 = [0.435,0.278,0.702]
select surf_pocket4, protein and id [7322,7334,2237,4519,7360,2830,2831,2832,2239,2257,2261,7367,7370,7373,7375,4465,4416,4521,2890,2894,2826,2898,2876,2870,7326] 
set surface_color,  pcol4, surf_pocket4 
set_color pcol5 = [0.698,0.361,0.902]
select surf_pocket5, protein and id [6719,6722,6724,5117,6711,6712,6713,6731,6733,6367,6363,5293,5295,6407,6403,6470,6673,6677,5112,5115,5236] 
set surface_color,  pcol5, surf_pocket5 
set_color pcol6 = [0.651,0.278,0.702]
select surf_pocket6, protein and id [6651,6684,1637,6644,1811,6671,6673,6676,6678,1674,1749,1673,1690,1697,1693,2440,1688,1694,1684,1675,6744] 
set surface_color,  pcol6, surf_pocket6 
set_color pcol7 = [0.902,0.361,0.824]
select surf_pocket7, protein and id [840,844,835,836,849,852,854,531,875,881,460,583,585,586,590,593,512,6007,6019,5932,5935,5981,5999,6009,6018] 
set surface_color,  pcol7, surf_pocket7 
set_color pcol8 = [0.702,0.278,0.533]
select surf_pocket8, protein and id [1609,1740,3880,3883,3787,3790,3931,1663,1651,1665,1718,1721,1723,3952,1605,3998,4004] 
set surface_color,  pcol8, surf_pocket8 
set_color pcol9 = [0.902,0.361,0.545]
select surf_pocket9, protein and id [5012,5019,5034,6876,6931,6949,6951,4974,1949,1951,2016,1955] 
set surface_color,  pcol9, surf_pocket9 
set_color pcol10 = [0.702,0.278,0.318]
select surf_pocket10, protein and id [1114,5583,1119,5579,5575,5578,1427,1428,5562,1422,1123,1482,1486,1115,1436,1437,1439] 
set surface_color,  pcol10, surf_pocket10 
set_color pcol11 = [0.902,0.451,0.361]
select surf_pocket11, protein and id [3373,3199,3200,3204,3206,3366,3369,3196,3075,1194,1240,1244,1248,1209,1234,1195,3131,1251] 
set surface_color,  pcol11, surf_pocket11 
set_color pcol12 = [0.702,0.459,0.278]
select surf_pocket12, protein and id [3663,3638,3657,3661,3645,2519,2468,2495,3641,3707,3805,3810,2498,2500,3644] 
set surface_color,  pcol12, surf_pocket12 
set_color pcol13 = [0.902,0.729,0.361]
select surf_pocket13, protein and id [2898,2900,2916,7246,2960,7244,2956,2958,2975,346,3274,3277] 
set surface_color,  pcol13, surf_pocket13 
set_color pcol14 = [0.702,0.675,0.278]
select surf_pocket14, protein and id [2827,2239,2259,2234,4416,4417,4455] 
set surface_color,  pcol14, surf_pocket14 
   

deselect

orient
