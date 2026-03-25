/**
 *******************************************************************************
 * @file     : keypad.c
 * @describe : Read keypad value and transform keypad value into type of command that user control mode.
 * 
 * @author : Woody.
 * @verstion : 0.1.0.
 * @date 20210803 0.1.0 Woody.
 *******************************************************************************
*/
#include "../../../System_Fuction/AVM/system.h"
#include "user_command.h"
#include "keypad.h"

#include <unistd.h>
#include <linux/input.h>
#include <fcntl.h>

#define DEBUG_MSG_KEYPAD                    // Print msg on debug mode.


#define DEV_PATH "/dev/input/event1"        // Depending on embedded systems, so difference is possible.
#include <pthread.h>


typedef enum                                // Value that determine transform or not.
{
    DETERMINE_TRANS_INIT,                   // Default value.
    DETERMINE_TRANS_EXECUTE,                // Execute transform.
    DETERMINE_TRANS_NON,                    // Don't need transform keypad value.
} detemine_trans_t;


typedef struct _keypad_transform_t          // Struct that get keypad value and transform.
{
    char *            devPath;        // Address of device path.
    int                     keysFd;         // Number of device.
    int                     keypadValue;    // Get keypad value.
    detemine_trans_t        determineTrans; // Determine transform or not.
    user_command_mode_t *   userCommand;    // Address of command that user control mode.
} keypad_transform_t;

/**
 * Function :
 *  Transform keypad value into type of command that user control mode.
 *
 * Parameter :
 *  detemine_trans_t * determineTrans :
 *      Determine that transform keypad value into command that user control mode.
 *      Value of determineTrans :
 *          DETERMINE_TRANS_INIT : Default value.
 *          DETERMINE_TRANS_EXECUTE : Execute transform.
 *          DETERMINE_TRANS_NON : Don't need transform keypad value.
 *
 *  Intput :
 *      int keypadvalueTmp : Value of keypad.
 *  Output :
 *      user_command_mode_t * userCommand : Address of command that user control mode.
*/
static void keypad_trans_user_command(   int                     keypadvalueTmp,
                                        detemine_trans_t *      determineTrans,
                                        user_command_mode_t *   userCommand)
{
    if(*determineTrans == DETERMINE_TRANS_EXECUTE)
    {
        switch(keypadvalueTmp)
        {
            case KEY_KPPLUS: *userCommand = CMD_PLUS; break;
            case KEY_KPMINUS: *userCommand = CMD_MINUS; break;
            case KEY_KPASTERISK: *userCommand = CMD_STAR; break;
            case KEY_KPENTER: *userCommand = CMD_ENTER; break;
            case KEY_KPDOT: *userCommand = CMD_DOT; break;
           // case KEY_KP8: *userCommand = CMD_UP; break;
           // case KEY_KP2: *userCommand = CMD_DOWN; break;
           // case KEY_KP4: *userCommand = CMD_LEFT; break;
           // case KEY_KP6: *userCommand = CMD_RIGHT; break;
            case KEY_KPSLASH: *userCommand = CMD_DIVIDE; break;
            case KEY_BACKSPACE: *userCommand = CMD_RETURN; break;

            case KEY_KP1: *userCommand = CMD_1; break;
            case KEY_KP2: *userCommand = CMD_2; break;
            case KEY_KP3: *userCommand = CMD_3; break;
            case KEY_KP4: *userCommand = CMD_4; break;
            case KEY_KP5: *userCommand = CMD_5; break;
            case KEY_KP6: *userCommand = CMD_6; break;
            case KEY_KP7: *userCommand = CMD_7; break;
            case KEY_KP8: *userCommand = CMD_8; break;
            case KEY_KP9: *userCommand = CMD_9; break;
            case KEY_KP0: *userCommand = CMD_0; break;

            case KEY_Q: *userCommand = CMD_Q; break;
            case KEY_W: *userCommand = CMD_W; break;
            case KEY_E: *userCommand = CMD_E; break;
            case KEY_R: *userCommand = CMD_R; break;
            case KEY_T: *userCommand = CMD_T; break;
            case KEY_Y: *userCommand = CMD_Y; break;
            case KEY_U: *userCommand = CMD_U; break;
            case KEY_I: *userCommand = CMD_I; break;
            case KEY_O: *userCommand = CMD_O; break;
            case KEY_P: *userCommand = CMD_P; break;

            case KEY_A: *userCommand = CMD_A; break;
            case KEY_S: *userCommand = CMD_S; break;
            case KEY_D: *userCommand = CMD_D; break;
            case KEY_F: *userCommand = CMD_F; break;
            case KEY_G: *userCommand = CMD_G; break;
            case KEY_H: *userCommand = CMD_H; break;
            case KEY_J: *userCommand = CMD_J; break;
            case KEY_K: *userCommand = CMD_K; break;
            case KEY_L: *userCommand = CMD_L; break;

            case KEY_Z: *userCommand = CMD_Z; break;
            case KEY_X: *userCommand = CMD_X; break;
            case KEY_C: *userCommand = CMD_C; break;
            case KEY_V: *userCommand = CMD_V; break;
            case KEY_B: *userCommand = CMD_B; break;
            case KEY_N: *userCommand = CMD_N; break;
            case KEY_M: *userCommand = CMD_M; break;
            case KEY_COMMA: *userCommand = CMD_COMMA; break;
            case KEY_DOT: *userCommand = CMD_DOT; break;
            case KEY_SLASH: *userCommand = CMD_SLASH; break;


            case KEY_SEMICOLON: *userCommand = CMD_SEMICOLON; break;

            default: *userCommand = CMD_DEFAULT;
        }

        *determineTrans = DETERMINE_TRANS_INIT;
    }
    // AVM_LOGI("userCommand %d \n",*userCommand);
}

/**
 * Function :
 *  Initialize and open control device.
 *
 * Parameter :
 *  intput :
 *      char * devPathTmp : Address of device path name.
 *
 * Return :
 *  Open device successfully return number of device.
 *  Open device unsuccessfully return -1.
*/
static int init_control_dev(char * devPathTmp)
{
    int keysFd;

    keysFd = open(devPathTmp, O_RDONLY);
    if (keysFd <= 0)
    {
        #ifdef DEBUG_MSG_KEYPAD
            AVM_LOGI("open /dev/input/event2 device error!\n");
        #endif

        return -1;
    }
    else
    {
        #ifdef DEBUG_MSG_KEYPAD
            AVM_LOGI("init_control_dev finish\n");
        #endif

        return keysFd;
    }
}

/**
 * Function :
 *  Get keypad value.
 *
 * Parameter :
 *  detemine_trans_t * determineTrans :
 *      Determine that transform keypad value into command that user control mode.
 *      When get keypad value finished, this funtionn only let determineTrans = DETERMINE_TRANS_EXECUTE.
 *      determineTrans is no anther mode in this funtion now.
 *
 *  Intput :
 *      int keysFd : Number of device.
 *  Output :
 *      int keypadvalue : address of keypad value. : Address of command that user control mode.
*/
static void keypad_read(int                 keysFd,
                        int *               keypadValue,
                        detemine_trans_t *  determineTrans)
{
    struct input_event t;
    static int bounce_flag;

    if(read(keysFd, &t, sizeof(t)) == sizeof(t))
    {
        if (t.type == EV_KEY)
        {
            if (t.value == 1)
            {
                if(t.code !=69)
                {
                    bounce_flag = 0;

                    *keypadValue = t.code;

                    if(*keypadValue != -1)
                    {
                        *determineTrans = DETERMINE_TRANS_EXECUTE;
                    }
                }
                #ifdef DEBUG_MSG_KEYPAD
                    AVM_LOGI("keypadValue %d \n",*keypadValue);
                #endif
            }
            else if(t.value == 0)
            {
                bounce_flag = 1;
                *determineTrans = DETERMINE_TRANS_EXECUTE;

                *keypadValue = -1;
            }
            // if(t.value == 0 && bounce_flag  == 0)
            // {
            //     bounce_flag = 1;

            //     *keypadValue = -1;
            // }
        }
    }
//    return 1;
}

/**
 * Function :
 *  A thread that read keypad value then transform keypad value into type of command that user control mode.
 *
 * Parameter :
 *  user_command_mode_t * userCommand : Address of command that user control mode.
*/
static void * keypad_read_trans_thread(void * userCommand)
{
    pthread_detach(pthread_self());
    static keypad_transform_t keypadTrans;

    user_command_mode_t* userCommandTmp;
    userCommandTmp = (user_command_mode_t *)userCommand;

    char devPath[] = DEV_PATH;
    keypadTrans.devPath = devPath;

    keypadTrans.keypadValue = -1;
    keypadTrans.determineTrans = DETERMINE_TRANS_INIT;

    keypadTrans.userCommand = userCommandTmp;

    keypadTrans.keysFd = init_control_dev(keypadTrans.devPath);
    if(keypadTrans.keysFd == -1)
    {
        AVM_LOGI("pthread_exit!\n");
        pthread_exit(0);
    }
    while (1)
    {
        usleep(50);
        keypad_read(keypadTrans.keysFd, &keypadTrans.keypadValue, &keypadTrans.determineTrans);

        keypad_trans_user_command(keypadTrans.keypadValue, &keypadTrans.determineTrans, keypadTrans.userCommand);
    }
}

void keypad_read_trans_thread_start(user_command_mode_t * userCommand)
{
    pthread_t threadKeypad;

    pthread_create(&threadKeypad, NULL, &keypad_read_trans_thread, userCommand);
}