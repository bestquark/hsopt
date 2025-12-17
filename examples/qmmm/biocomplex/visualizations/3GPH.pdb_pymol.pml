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

load "data/3GPH.pdb", protein
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

load "data/3GPH.pdb_points.pdb.gz", points
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
select surf_pocket1, protein and id [2231,2201,2202,2204,2205,1408,2193,2203,2220,2230,1406,1403,1405,3336,3340,3341,3334,3335,3342,2721,3271,2725,2726,2728,3285,3276,3281,3283,3272,840,2177,2237,3366,3367,3370,2227,2232,2233,2679,2680,2682,3405,2257,841,1200,3365,3397,2206,3326,3331,680,3315,3330,678,686,688,1741,2195,2194,2196,2197,1437,1715,1434,2733,2734,2735,1442,1443,1444,1446,3654,1448,3277,3278,3290,3291,2959,2781,561,2755,2761,2763,2764,3655,1473,604,3319,676,560,682,684,1740,679,638] 
set surface_color,  pcol1, surf_pocket1 
set_color pcol2 = [0.278,0.361,0.702]
select surf_pocket2, protein and id [5958,5968,5970,5971,5972,5969,5991,5993,5995,5996,5997,5960,5961,5986,5984,5174,5987,5169,5171,7097,7100,7102,7106,7104,7047,7049,7101,7107,7108,7110,4448,6530,4452,4453,4454,6529,7420,7421,5963,4378,4379,5964,5239,5962,5200,5203,7081,7092,4465,7085,4333,4334,6527,7082,5994,6494,5209,5210,5212,5213,5214,5208,7056,7044,6725,7037,7038,7051,6501,6499,7057,6521,6500,7133,7136,5998,5999,7131,7132,7163,5974,6446,6449,7171,7172,6023,5943,5932,5931,5939,5505,5506,5507,4606,4966,5481,4445,4444,4446] 
set surface_color,  pcol2, surf_pocket2 
set_color pcol3 = [0.373,0.361,0.902]
select surf_pocket3, protein and id [2766,2769,2770,2923,1550,571,1551,1553,1561,355,2771,2752,583,1474,3651,1470,3649,213,214,376,138,180,181,377,1560,338,2937,2938,2746,2747,2748,3644,3646,1524,1525,572] 
set surface_color,  pcol3, surf_pocket3 
set_color pcol4 = [0.380,0.278,0.702]
select surf_pocket4, protein and id [7417,5291,5290,7410,7412,7414,6518,6513,3953,3954,6514,4356,6535,6536,6537,4111,5325,5326,5327,3911,6704,3987,4149,4150,4344,5315,5316,4128] 
set surface_color,  pcol4, surf_pocket4 
set_color pcol5 = [0.600,0.361,0.902]
select surf_pocket5, protein and id [6464,6470,6469,6471,7380,7363,7372,7369,6730,6760,6762,6764,6766,6838,6839,6485,6473,7431,7379,3962,3963,3966,3959,3945,7406,6735,6736,6729,6734,3993,3994,6731,6732] 
set surface_color,  pcol5, surf_pocket5 
set_color pcol6 = [0.557,0.278,0.702]
select surf_pocket6, protein and id [2964,2994,2995,2963,2965,2966,2969,2996,2998,2719,3613,3640,220,2970,221,184,185,189,190,193,2968,186,172,173,3000,3070,3073,3072,2698,2707,2288,3614,3597,3603,3606,2704,2703,2705] 
set surface_color,  pcol6, surf_pocket6 
set_color pcol7 = [0.831,0.361,0.902]
select surf_pocket7, protein and id [5215,5219,5185,5186,7462,5211,7439,7438,7443,7459,5182,5184,4887,7455,6040,6041,4899,4888,4897] 
set surface_color,  pcol7, surf_pocket7 
set_color pcol8 = [0.702,0.278,0.667]
select surf_pocket8, protein and id [1445,2250,1416,1418,1419,1420,3694,3672,3677,2223,1121,2274,2275,3689,3693,3695,3696,1122,1453,1442,3673,1455,1457] 
set surface_color,  pcol8, surf_pocket8 
set_color pcol9 = [0.902,0.361,0.741]
select surf_pocket9, protein and id [7145,7144,6329,6330,6393,6422,6423,6448,6412,6965,6394,7152,7142,7143,6331,7138,7140,7141,7121] 
set surface_color,  pcol9, surf_pocket9 
set_color pcol10 = [0.702,0.278,0.486]
select surf_pocket10, protein and id [6576,6577,1987,6584,1955,1907,1908,826,857,860,803,805,832,833,869,1956,1978] 
set surface_color,  pcol10, surf_pocket10 
set_color pcol11 = [0.902,0.361,0.514]
select surf_pocket11, protein and id [845,842,843,847,836,837,838,3356,3358,841,1221,3362,3355,3354,3339,3342,3389,906,874] 
set surface_color,  pcol11, surf_pocket11 
set_color pcol12 = [0.702,0.278,0.310]
select surf_pocket12, protein and id [1474,3651,1470,3649,583,580,1550,1524,1525,603,595] 
set surface_color,  pcol12, surf_pocket12 
set_color pcol13 = [0.902,0.435,0.361]
select surf_pocket13, protein and id [2536,2563,2564,2565,3373,3374,3382,3383,2628,3199,2538,3389,3386,3377,3378,3379,2656,2657,3216,3357,3375] 
set surface_color,  pcol13, surf_pocket13 
set_color pcol14 = [0.702,0.427,0.278]
select surf_pocket14, protein and id [6279,6283,6284,6292,4742,4745,4777,4710,4741,4740,7218,7242,7251,7252,7253] 
set surface_color,  pcol14, surf_pocket14 
set_color pcol15 = [0.902,0.667,0.361]
select surf_pocket15, protein and id [4176,4184,4187,6774,6738,4018,4179,4181,6736,3991,3994,6759,4016,4017,4019,4052,4054,4055,4058,4059] 
set surface_color,  pcol15, surf_pocket15 
set_color pcol16 = [0.702,0.608,0.278]
select surf_pocket16, protein and id [406,408,2970,221,253,403,411,404,254,244,2972,3003,3008,2993] 
set surface_color,  pcol16, surf_pocket16 
set_color pcol17 = [0.902,0.894,0.361]
select surf_pocket17, protein and id [5604,5683,5015,5605,5630,5591,5594,5026,4999,5022,5024] 
set surface_color,  pcol17, surf_pocket17 
   

deselect

orient
