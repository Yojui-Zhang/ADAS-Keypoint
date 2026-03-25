/**
 *******************************************************************************
 * @file : keypad_control.c
 * @describe : .
 *
 * @author : Woody
 * @verstion : 0.1.0.
 * @date 20201105 0.1.0 Woody.
 * @date 20201222 0.1.1 Linda.
 *******************************************************************************
*///
// Created by AI611 on 2021/11/5.
//

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>

#include "system.h"
#include "ui_struct.h"
#include "include/ui.h"
#include "keypad_control.h"
#include "fp_source.h"
int hh= 0;
void control_camara_parameter(user_command_mode_t user_command, camera_para_t * para)
{
	switch(user_command)
	{
		case CMD_Q:
			para->exPara.tmpCenter[0] += 0.1;
			AVM_LOGI("TmpCenter[0] = %f \n",para->exPara.tmpCenter[0]);
			break;
		case CMD_A:
			para->exPara.tmpCenter[0] -= 0.1;
			AVM_LOGI("TmpCenter[0] = %f \n",para->exPara.tmpCenter[0]);
			break;
		case CMD_W:
			para->exPara.tmpCenter[1] += 0.1;
			AVM_LOGI("TmpCenter[1] = %f \n",para->exPara.tmpCenter[1]);
			break;
		case CMD_S:
			para->exPara.tmpCenter[1] -= 0.1;
			AVM_LOGI("TmpCenter[1] = %f \n",para->exPara.tmpCenter[1]);
			break;
		case CMD_E:
			para->exPara.tmpCenter[2] += 0.1;
			AVM_LOGI("TmpCenter[2] = %f \n",para->exPara.tmpCenter[2]);
			break;
		case CMD_D:
			para->exPara.tmpCenter[2] -= 0.1;
			AVM_LOGI("TmpCenter[2] = %f \n",para->exPara.tmpCenter[2]);
			break;
		case CMD_R:
			para->exPara.tmpCenter[3] += 0.1;
			AVM_LOGI("TmpCenter[3] = %f \n",para->exPara.tmpCenter[3]);
			break;
		case CMD_F:
			para->exPara.tmpCenter[3] -= 0.1;
			AVM_LOGI("TmpCenter[3] = %f \n",para->exPara.tmpCenter[3]);
			break;
		case CMD_T:
			para->exPara.tmpCenter[4] += 0.1;
			AVM_LOGI("TmpCenter[4] = %f \n",para->exPara.tmpCenter[4]);
			break;
		case CMD_G:
			para->exPara.tmpCenter[4] -= 0.1;
			AVM_LOGI("TmpCenter[4] = %f \n",para->exPara.tmpCenter[4]);
			break;
		case CMD_Y:
			para->exPara.tmpCenter[5] += 0.1;
			AVM_LOGI("TmpCenter[5] = %f \n",para->exPara.tmpCenter[5]);
			break;
		case CMD_H:
			para->exPara.tmpCenter[5] -= 0.1;
			AVM_LOGI("TmpCenter[5] = %f \n",para->exPara.tmpCenter[5]);
			break;
		case CMD_U:
			para->exPara.tmpCenter[6] += 0.1;
			AVM_LOGI("TmpCenter[6] = %f \n",para->exPara.tmpCenter[6]);
			break;
		case CMD_J:
			para->exPara.tmpCenter[6] -= 0.1;
			AVM_LOGI("TmpCenter[6] = %f \n",para->exPara.tmpCenter[6]);
			break;
		case CMD_I:
			para->exPara.tmpCenter[7] += 0.1;
			AVM_LOGI("TmpCenter[7] = %f \n",para->exPara.tmpCenter[7]);
			break;
		case CMD_K:
			para->exPara.tmpCenter[7] -= 0.1;
			AVM_LOGI("TmpCenter[7] = %f \n",para->exPara.tmpCenter[7]);
			break;
		case CMD_O:
			para->exPara.tmpCenter[8] += 0.1;
			AVM_LOGI("TmpCenter[8] = %f \n",para->exPara.tmpCenter[8]);
			break;
		case CMD_L:
			para->exPara.tmpCenter[8] -= 0.1;
			AVM_LOGI("TmpCenter[8] = %f \n",para->exPara.tmpCenter[8]);
			break;
		case CMD_P:
			para->exPara.tmpCenter[9] += 0.1;
			AVM_LOGI("TmpCenter[9] = %f \n",para->exPara.tmpCenter[9]);
			break;
		case CMD_SEMICOLON:
			para->exPara.tmpCenter[9] -= 0.1;
			AVM_LOGI("TmpCenter[9] = %f \n",para->exPara.tmpCenter[9]);
			break;
		case CMD_Z:
			para->exPara.tmpCenter[10] += 0.1;
			AVM_LOGI("TmpCenter[10] = %f \n",para->exPara.tmpCenter[10]);
			break;
		case CMD_X:
			para->exPara.tmpCenter[10] -= 0.1;
			AVM_LOGI("TmpCenter[10] = %f \n",para->exPara.tmpCenter[10]);
			break;
		case CMD_C:
			para->exPara.tmpCenter[11] += 0.1;
			AVM_LOGI("TmpCenter[11] = %f \n",para->exPara.tmpCenter[11]);
			break;
		case CMD_V:
			para->exPara.tmpCenter[11] -= 0.1;
			AVM_LOGI("TmpCenter[11] = %f \n",para->exPara.tmpCenter[11]);
			break;
		case CMD_B:
			para->exPara.tmpCenter[12] += 0.1;
			AVM_LOGI("TmpCenter[12] = %f \n",para->exPara.tmpCenter[12]);
			break;
		case CMD_N:
			para->exPara.tmpCenter[12] -= 0.1;
			AVM_LOGI("TmpCenter[12] = %f \n",para->exPara.tmpCenter[12]);
			break;
		case CMD_M:
			para->exPara.tmpCenter[13] += 0.1;
			AVM_LOGI("TmpCenter[13] = %f \n",para->exPara.tmpCenter[13]);
			break;
		case CMD_COMMA:
			para->exPara.tmpCenter[13] -= 0.1;
			AVM_LOGI("TmpCenter[13] = %f \n",para->exPara.tmpCenter[13]);
			break;
		case CMD_DOT:
			para->exPara.tmpCenter[14] += 0.1;
			AVM_LOGI("TmpCenter[14] = %f \n",para->exPara.tmpCenter[14]);
			break;
		case CMD_SLASH:
			para->exPara.tmpCenter[14] -= 0.1;
			AVM_LOGI("TmpCenter[14] = %f \n",para->exPara.tmpCenter[14]);
			break;
		case CMD_7:
			para->exPara.phi += 0.1;
			AVM_LOGI("phi = %f \n",para->exPara.phi);
			break;
		case CMD_4:
			para->exPara.phi -= 0.1;
			AVM_LOGI("phi = %f \n",para->exPara.phi);
			break;
		case CMD_8:
			para->exPara.theta += 1;
			AVM_LOGI("theta = %f \n",para->exPara.theta);
			break;
		case CMD_5:
			para->exPara.theta -= 1;
			AVM_LOGI("theta = %f \n",para->exPara.theta);
			break;
		case CMD_9:
			para->exPara.eyeRadius += 0.1;
			AVM_LOGI("eyeRadius = %f \n",para->exPara.eyeRadius);
			break;
		case CMD_6:
			para->exPara.eyeRadius -= 0.1;
			AVM_LOGI("eyeRadius = %f \n",para->exPara.eyeRadius);
			break;
		case CMD_1:
			hh += 1;
			AVM_LOGI("hh = %d \n",hh);
			break;
		case CMD_2:
			hh -= 1;
			AVM_LOGI("hh = %d \n",hh);
			break;
		case CMD_ENTER:
			break;
		default:
			break;
	}
	save_para_info(para);
}

extern int recordstartFlag;
int recordtime = 0;
extern int frame;
extern avm_ui_info_t avmUiInfo;

FILE * fp_record;
struct timeval recordStart, recordend;

void control_record(user_command_mode_t *user_command, can_bus_info_t canBus, can_bus_info_t canBusOld, avm_view_t view)
{
	float eye_radius_tmp;

	// if(avmUiInfo.view == VIEW_RECORD_4_IN_1)
	// // if(avmUiInfo.view == VIEW_START)
	// {
	// 	if(*user_command == CMD_ENTER)
	// 	{
	// 		if(recordstartFlag == 1)
	// 		{
	// 			recordstartFlag = 0;

	// 			fclose(fp_record);
	// 		}
	// 		else if(recordstartFlag == 0)
	// 		{
	// 			recordstartFlag = 1;
	// 			recordtime++;
				char filename[50];
				sprintf(filename, "./data/record/record_%d.txt",recordtime+2);
				fp_record = fopen(filename,"a");

				// gettimeofday(&recordStart, NULL);

				// if (fp_record == NULL)
				// {
				// 	AVM_LOGE("Error opening file!\n");
				// 	exit(1);
				// }
				// fprintf(fp_record, "start\n");
		// 	}  
		// }
		// cleanCommendFlag = 1;

		// if(recordstartFlag)
		// {
			// gettimeofday(&recordend, NULL);
			// float temp_time = calculate_time(recordStart, recordend);
			float temp_time = 0.0;

			// fprintf(fp_record, "gear:%c gearold:%c keymode:%d steer:%-7.3lf Turning: %c frame %d time %fs view %d\n", canBus.gear,canBusOld.gear, canBus.keymode, canBus.steer, canBus.turningSignal, frame, temp_time, view);
		// }
	// }
			fclose(fp_record);
}
