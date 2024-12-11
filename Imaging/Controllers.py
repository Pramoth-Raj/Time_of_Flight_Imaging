from serial import Serial # type: ignore
from TimeTagger import createTimeTagger, Histogram # type: ignore
import numpy as np # type: ignore

class MotorController:

    def __init__(self, com_port: str, baud_rate=9600) -> None:
        '''
        Constructor for MotorController Class
            Parameters:
                com_port (str): Port in which the arduino is connected
                baud_rate (int): Data rate for commmunication with arduino
            Returns:
                object (MotorController): MotorController object 
        '''
        self.com_port = com_port
        self.baud_rate = baud_rate
        self.ser = Serial(com_port, baudrate=self.baud_rate)
        self.ser.readline();
        self.ser.readline();
        self.ser.readline();

    def send_command(self,command):
        self.ser.write(command.encode())
    
    def home_set(self):
        self.send_command('X')
        print(self.ser.readline())

    def right_step(self,dist,speed):
        self.send_command(f'R{dist} {speed}')
        print(self.ser.readline())

    def left_step(self,dist,speed):
        self.send_command(f'L{dist} {speed}')
        print(self.ser.readline())
    
    def down_step(self,dist,speed):
        self.send_command(f'D{dist} {speed}')
        print(self.ser.readline())

    def up_step(self,dist,speed):
        self.send_command(f'U{dist} {speed}')
        print(self.ser.readline())

    def x_step(self,dist, speed, xdirection):
        dist*=6400
        speed*=6400
        if xdirection == 0:
            self.left_step(dist,speed)
        elif xdirection == 1:
            self.right_step(dist,speed)
        else:
            exit()    

    def y_step(self,dist, speed, ydirection):
        dist*=6400
        speed*=6400
        if ydirection == 0:
            self.up_step(dist,speed)
        elif ydirection == 1:
            self.down_step(dist,speed)
        else:
            exit()

    def home_step(self):
        self.send_command('H')  
        print(self.ser.readline())
        print(self.ser.readline())
        print(self.ser.readline())

    def getlocation(self):
        self.send_command('E')
        print(self.ser.readline())
        t =  int(self.ser.readline().split()[-1]), int(self.ser.readline().split()[-1])
        #print(t)
        return t[0]/6400,t[1]/6400
    

class TTController:

    def __init__(self, spad_channel=2, trigger_channel=1, spad_ch_delay=0.0, trigger_ch_delay=1.072e6):
        self.spad_channel = spad_channel
        self.trigger_channel = trigger_channel
        self.spad_ch_delay = spad_ch_delay
        self.trigger_ch_delay = trigger_ch_delay
        self.tagger = createTimeTagger()

    def acquire_histogram(self, acquisition_time=1, bin_width=10e-12, nbins=500):
        bin_width_ps = bin_width*1e12
        t=acquisition_time*1e12 #every time data converted to ps
        hist = Histogram(self.tagger, self.spad_channel, self.trigger_channel, bin_width_ps, nbins)
        hist.startFor(t) #60e12 #in ps
        while hist.isRunning():
            continue    
        arr = np.array(hist.getData())
        return arr # returns x in picosecond
    

class MasterController(MotorController, TTController):

    def __init__(self, com_port, baud_rate=9600, spad_channel=2, trigger_channel=1, spad_ch_delay=0, trigger_ch_delay=1.072e6):
        '''
        Constructor for MotorController Class
            Parameters:
                com_port (str): Port in which the arduino is connected
                baud_rate (int): Data rate for commmunication with arduino
            Returns:
                object (MotorController): MotorController object 
        '''
        MotorController.__init__(self, com_port, baud_rate)
        TTController.__init__(self, spad_channel, trigger_channel, spad_ch_delay, trigger_ch_delay)

