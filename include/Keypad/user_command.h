/**
 *******************************************************************************
 * @file     : keypad.h
 * @describe : This file contains the user control command defines of the application.
 * 
 * @author : Woody.
 * @verstion : 0.1.0.
 * @date 20210803 0.1.0 Woody.
 * @date 20211220 0.1.1 Linda.
 *******************************************************************************
*/
#ifndef _USER_COMMAND_H_
#define _USER_COMMAND_H_


typedef enum //Type of command that user control mode.        
{
    CMD_DEFAULT = -1,
    CMD_NONE = 0,
    CMD_0,
    CMD_1,
    CMD_2,
    CMD_3,
    CMD_4,
    CMD_5,
    CMD_6,
    CMD_7,
    CMD_8,
    CMD_9,          //10
    CMD_PLUS,
    CMD_MINUS,
    CMD_STAR,
    CMD_ENTER,
    CMD_DOT,
    CMD_UP,
    CMD_DOWN,
    CMD_LEFT,
    CMD_RIGHT,
    CMD_DIVIDE,     //20
    CMD_RETURN,
    CMD_MENU,
    CMD_POWER,
    CMD_3D,
    CMD_ROTATE,

    CMD_Q,
    CMD_W,
    CMD_E,
    CMD_R,
    CMD_A,          //30
    CMD_S,
    CMD_D,
    CMD_F,
    CMD_Z,
    CMD_X,
    CMD_C,
    CMD_V,

    CMD_T,
    CMD_Y,          //40
    CMD_U,
    CMD_I,
    CMD_G,
    CMD_H,
    CMD_J,
    CMD_K,
    CMD_B,
    CMD_N,
    CMD_M,
    CMD_COMMA,      //50
    CMD_SLASH,

    CMD_O,
    CMD_L,
    CMD_P,
    CMD_SEMICOLON,
    CMD_ACC_RESUME,

    CMD_KEYPAD_LIGHT_GREEN,
    CMD_KEYPAD_LIGHT_ORANGE,
    CMD_KEYPAD_LIGHT_RED,
    CMD_KEYPAD_LIGHT_CLEAR,

    CMD_KEYPAD_SIGN_110,
    CMD_KEYPAD_SIGN_30,
    CMD_KEYPAD_SIGN_40,
    CMD_KEYPAD_SIGN_50,
    CMD_KEYPAD_SIGN_60,
    CMD_KEYPAD_SIGN_70,
    CMD_KEYPAD_SIGN_80,
    CMD_KEYPAD_SIGN_90,
    CMD_KEYPAD_SIGN_100,
} user_command_mode_t;

#endif
