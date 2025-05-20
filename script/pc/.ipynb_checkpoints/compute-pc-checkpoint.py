import numpy as np                          # type: ignore
import sys      
sys.path.insert(1, "/home/users/mendrika/SSA/SA/module/")
import snflics                               # type: ignore

hour = sys.argv[1]
filter_size = int(sys.argv[2])


minutes = ["00", "15", "30", "45"]

# These are the bounds for Zambia
ymin, ymax = 578, 927
xmin, xmax = 1471, 1859


for minute in minutes:
    data_path = "/gws/nopw/j04/cocoon/SSA_domain/ch9_wavelet/"
    dataset = snflics.search(hour, minute, data_path=data_path)
    pc = snflics.compute_pc(dataset, filter_size, ymin, ymax, xmin, xmax)
    np.save(f"/gws/nopw/j04/wiser_ewsa/mrakotomanga/Zambia/output/pc-Zambia/diurnal/pc-Zambia-{hour}-{minute}-{filter_size}.npy", pc)
