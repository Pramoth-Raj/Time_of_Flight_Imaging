import numpy as np 
import matplotlib.pyplot as plt 
import time
from .Controllers import MasterController

class Scanner:

    def __init__(self, master_controller: MasterController) -> None:
        self.master_controller = master_controller

    def scan_2D(self, delay=0.05, rep_rate=10, acquisition_time=1, bin_width=10e-12, nbins=500, start_direction=(1,0), resolution=(0.1, 0.1), area=(15, 5)):
        xdirection, ydirection = start_direction
        total_delay = delay + acquisition_time
        x_range, y_range = area
        x_resolution, y_resolution = resolution
        num_xsteps = int(x_range/x_resolution)
        num_ysteps = int(y_range/y_resolution)
        stepper_speed_x = x_resolution/(total_delay)
        stepper_speed_y = 1

        self.master_controller.home_set()

        start_time = time.time()

        time_delay_2d = []
        for j in range(num_ysteps):
            self.master_controller.x_step(x_range, stepper_speed_x, xdirection)
            print("command given to move in x axis")
            time_array = np.arange(total_delay, total_delay*(num_xsteps+1), total_delay)
            time_delay_1d = []
            for i in range(num_xsteps):
                # if i%100==0:
                #     print(i)
                while time.time()<=start_time + time_array[i]:
                    continue
                
                time_delay_1d.append(self.master_controller.acquire_histogram(acquisition_time,bin_width,nbins))
            time_delay_2d.append(time_delay_1d)

            while (abs(self.master_controller.getlocation()[0]) < x_range and xdirection) or (self.master_controller.getlocation()[0]<0 and not xdirection):
                print(self.master_controller.getlocation())
                continue
            self.master_controller.y_step(y_resolution,stepper_speed_y, ydirection)

            while (abs(self.master_controller.getlocation()[1]) < round(y_resolution*(j+1) , 3)):
                continue
            xdirection = not(xdirection)
            print("one iteration of for loop completed")
            
        self.master_controller.home_step()

        return time_delay_2d

    def scan_1D(self, delay=0.05, rep_rate=10, acquisition_time=1, bin_width=10e-12, nbins=500, start_direction=(1,0), resolution=(0.1, 0.1), area=(15, 5)):
        xdirection = start_direction[0] if isinstance(start_direction, tuple) else start_direction
        total_delay = delay + acquisition_time
        x_range = area[0] if isinstance(area, tuple) else area
        x_resolution = resolution[0] if isinstance(resolution, tuple) else resolution
        num_xsteps = int(x_range/x_resolution)
        stepper_speed = x_resolution/total_delay

        self.master_controller.home_set()

        start_time = time.time()

        self.master_controller.x_step(x_range, stepper_speed, xdirection)
        time_array = np.arange(total_delay, total_delay*(num_xsteps+1), total_delay)
        time_delay_1d = []

        for i in range(num_xsteps):
                # if i%100==0:
                #     print(i)
            while time.time()<=start_time + time_array[i]:
                continue
                
            time_delay_1d.append(self.master_controller.acquire_histogram(acquisition_time,bin_width,nbins))

        while (abs(self.master_controller.getlocation()[0]) < x_range and xdirection) or (self.master_controller.getlocation()[0]<0 and not xdirection):
            print(self.master_controller.getlocation())
            continue

            # while (abs(getlocation()[0]) < x_dist and xdirection) or (getlocation()[0]<0 and not xdirection):
            #     print(getlocation())
            #     continue

        self.master_controller.home_step()


        return time_delay_1d
    

if __name__=='__main__':
    mc = MasterController("COM13")
    scanner = Scanner(mc)
    scanner.master_controller.home_set()
    scanner.master_controller.x_step(1,1,1)
    print(scanner.master_controller.getlocation())
    exit()
