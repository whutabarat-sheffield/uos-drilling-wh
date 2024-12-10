from enum import Enum
from collections import namedtuple

class SetitecStopCodes(namedtuple('SetitecStopCode','code description action'),Enum):
    NORMAL = 0x000000, 'No stop code: Machine keeps drilling', 'Keep Drilling'
    TORQUE_MAX = 0x000001, 'Detection of a torque rise', 'Go to next Step'
    TORQUE_LIMIT = 0x000002, 'Reaching the Torque Limit of the machine (70A)', 'STOP'
    THRUST_MAX = 0x000004, 'Detection of a thrust rise', 'Go to next Step'
    THRUST_LIMIT = 0x000008, 'Reaching the Thrust Limit of the machine (Hard Stop)', 'Spindle Retract'
    THRUST_SAFETY = 0x000010, 'Reaching the Thrust Safety defined', 'STOP'
    TORQUE_SAFETY = 0x000020, 'Reaching the Torque Safety defined', 'STOP'
    STROKE_LIMIT = 0x000040, 'Reach the maximum stroke of the step', 'Go to the next Step'
    STOP_BUTTON = 0x000080, 'Stop button pressed', 'STOP'
    STOP_CYCLE = 0x000100, 'Over Temp, Stop button or ARU detected', 'STOP'
    TOOL_BREAK = 0x000200, 'Tool Break detection function', 'STOP'
    ABORT = 0x000400, 'Blue Button pressed', 'Spindle Retract'
    TORQUE_MIN = 0x000800, 'Detection of a torque drop', 'Go to next Step'
    THRUST_MIN = 0x001000, 'Detection of a thrust drop', 'Go to next Step'
    STROKE_MAX = 0x002000, 'Reach the maximum stroke defined', 'Spindle Retract'
    STROKE_TORQUE_LIMIT = 0x004000, 'Material detection function with the Torque', 'Spindle Retract'
    STROKE_THRUST_LIMIT = 0x008000, 'Material detection function with the Thrust', 'Spindle Retract'
    KILLMOT_M2 = 0x010000, 'Feed motor short circuit detection', 'STOP'
    KILLMOT_M1 = 0x020000, 'Rotation motor short circuit detection', 'STOP'
    KILLMOT_M1_TORQUE_LIMIT = 0x020002, 'Rotation motor short circuit detection and reaching the Torque Limit of the machine (70A)', 'STOP'
    POWER_MAX = 0x040000, 'Maximum Power of the Controlled reached', 'STOP'
    DELAY = 0x080000, 'Delay of step passed', 'Go to next Step'
    DRIVE_OVER_TEMP = 0x100000, 'Drive max Temperature reached (80 C)', 'STOP'
    TOOL_OVER_TEMP = 0x200000, 'Tool max Temperature reached (65 C)', 'STOP'
    OVERLOAD_M1 = 0x400000, 'Overload of the Drive for the M1 Motor', 'STOP'
    OVERLOAD_M2 = 0x800000, 'Overload of the Drive for the M2 Motor', 'STOP'

    def match(target):
        ''' search for a code that contains the target '''
        for v in SetitecStopCodes:
            if target in v:
                return v
