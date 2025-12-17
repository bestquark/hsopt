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

load "data/3GPH_chainA.pdb", protein
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

load "data/3GPH_chainA.pdb_points.pdb.gz", points
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
select surf_pocket1, protein and id [2188,2197,2187,604,2190,1398,1466,3647,3648,681,679,2756,2757,3329,3327,3319,3324,3308,3323,2714,2726,2727,2728,2748,3264,2952,3269,3270,3271,3283,3284,1435,1436,1437,1439,1441,2186,2189,1733,631,675,671,672,3333,673,2754,560,561,3309,2774,677,669,735,738,3312,1430,1427,1708,1734,2223,2220,2225,2226,2721,2224,2230,3276,3328,3363,3274,3278,3359,3360,3398,2250,2718,2719,3265,2672,2673,2675,2213,2194,2195,2198,2199,1401,2196,1399,3335,3358,3390,3334,1193,2170,834,833,1396] 
set surface_color,  pcol1, surf_pocket1 
set_color pcol2 = [0.278,0.341,0.702]
select surf_pocket2, protein and id [591,1543,1542,1544,1546,1548,571,572,2762,2916,354,355,1571,1572,1550,181,1553,1554,1513,138,377,213,2930,214,376,2931,1562,338,580,583,1467,3644,595,2759,2763,2764,2745,3637,1463,180,1517,1518,2739,2740,2741,575,603,3639,3642,1533,1520] 
set surface_color,  pcol2, surf_pocket2 
set_color pcol3 = [0.424,0.361,0.902]
select surf_pocket3, protein and id [2712,2988,2993,2958,2959,2961,2989,2991,2957,2987,2956,2962,2963,2965,2986,3606,3607,3633,173,190,193,220,219,221,172,184,185,186,189,408,253,254,2691,2697,2696,2698,3065,3063,3066,2700,2281,3599,3590,3596] 
set surface_color,  pcol3, surf_pocket3 
set_color pcol4 = [0.435,0.278,0.702]
select surf_pocket4, protein and id [3665,3670,1114,2267,3682,1411,1412,1413,1115,1124,1126,3686,3687,3688,3689,2243,2216,2268,1446,1448,1409,1450,1438,1435,3666] 
set surface_color,  pcol4, surf_pocket4 
set_color pcol5 = [0.698,0.361,0.902]
select surf_pocket5, protein and id [3370,3371,3372,2558,3366,3367,3368,3375,3376,3192,3190,2649,2650,3211,3209,3248,3348,3350,2556,2557,2621] 
set surface_color,  pcol5, surf_pocket5 
set_color pcol6 = [0.651,0.278,0.702]
select surf_pocket6, protein and id [3335,3355,812,3332,899,3347,3348,3349,3351,830,867,829,836,840,838,1214,835,839,834,831,3382] 
set surface_color,  pcol6, surf_pocket6 
set_color pcol7 = [0.902,0.361,0.824]
select surf_pocket7, protein and id [406,408,403,404,411,412,414,254,426,430,221,279,281,282,285,286,244,3001,3009,2963,2965,2986,2996,3003,3008] 
set surface_color,  pcol7, surf_pocket7 
set_color pcol8 = [0.702,0.278,0.533]
select surf_pocket8, protein and id [798,862,1948,1949,1900,1901,1971,825,819,826,850,851,853,1980,796,2005,2007] 
set surface_color,  pcol8, surf_pocket8 
set_color pcol9 = [0.902,0.361,0.545]
select surf_pocket9, protein and id [2510,2512,2519,3445,3469,3479,3480,2491,968,969,1003,972] 
set surface_color,  pcol9, surf_pocket9 
set_color pcol10 = [0.702,0.278,0.318]
select surf_pocket10, protein and id [545,2795,548,2793,2790,2792,707,708,2783,704,550,737,739,546,711,712,714] 
set surface_color,  pcol10, surf_pocket10 
set_color pcol11 = [0.902,0.451,0.361]
select surf_pocket11, protein and id [1689,1603,1604,1606,1607,1687,1688,1602,1537,587,611,613,615,594,607,588,1570,617] 
set surface_color,  pcol11, surf_pocket11 
set_color pcol12 = [0.702,0.459,0.278]
select surf_pocket12, protein and id [1832,1818,1829,1831,1821,1251,1226,1240,1819,1857,1908,1910,1241,1242,1820] 
set surface_color,  pcol12, surf_pocket12 
set_color pcol13 = [0.902,0.729,0.361]
select surf_pocket13, protein and id [1450,1451,1459,3625,1480,3624,1478,1479,1487,165,1643,1644] 
set surface_color,  pcol13, surf_pocket13 
set_color pcol14 = [0.702,0.675,0.278]
select surf_pocket14, protein and id [1410,1115,1125,1112,2216,2217,2237] 
set surface_color,  pcol14, surf_pocket14 
   

deselect

orient
