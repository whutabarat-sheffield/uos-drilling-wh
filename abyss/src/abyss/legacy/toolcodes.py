from enum import IntEnum

class SetiCodes(IntEnum):
    '''
        Status codes for Seti Electric Drill

        The value of the code is from the official manual
        Use a label attribute to access the official description
    '''
    def __new__(cls,value,descr):
        obj = int.__new__(cls,value)
        obj._value_ = value
        obj.label = descr
        return obj
    NORMAL_MODE = (0, "Normal mode")
    MAC_TORQUE = (1,  "Max. torque")
    STOP_LIMIT_TORQUE_70A = (2, "Stop limit torque - 70A")  
    MAX_THRUST = (4, "Max. thrust")
    STOP_LIMIT_THRUST = (8, "Stop limit thrust")
    SAFETY_THRUST = (16, "Safety thrust") 
    SAFETY_TORQUE = (32, "Safety torque") 
    STOP = (256, "Stop (red button)")
    TOOL_BREAK = (512, "Tool break") 
    ABORT = (1024, "Abort cycle (blue button)")
    MIN_THRUST_MIN = (2048, "Min. thrust min") 
    MIN_TORQUE_MIN = (4096, "Min. torque min")
    STROKE_LIMIT = (8192, "Stroke limit")
