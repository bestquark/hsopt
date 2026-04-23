# Taylor/LCU Dynamics

`taylor_lcu/h2/simple_taylor_lcu.py` estimates direct real-time dynamics with a
segmented truncated Taylor/LCU construction. Qualtran supplies sparse chemistry
SELECT/PREPARE counts; the example keeps the segment count, Taylor order,
amplification proxy, and coefficient precision as editable policy knobs.
