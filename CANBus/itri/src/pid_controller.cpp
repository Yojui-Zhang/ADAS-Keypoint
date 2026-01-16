#ifdef USE_ITRI_CAN
#include "unistd.h"


#include "pid_controller.h"
#include "canbus_recv.h"
#include "lib.h"
#include "terminal.h"
#include "pid_controller.h"

extern float target_speed;
extern volatile int steerCtrlMode;
extern double targetAngle; // left 0.0 ~ -510.0 , right 0.0 ~ 510.0 
extern double deceleration; // 0.0 - 10.0
extern float Targetdistance;

int stop_flag = 0;


CAR S3;

PID_incremental::PID_incremental():kp(0),ki(0),kd(0),e_pre_1(0),e_pre_2(0),target(0),actual(0)
{
   A=kp+ki+kd;
   B=-2*kd-kp;
   C=kd;
   e=target-actual;
}
PID_incremental::PID_incremental(float p,float i,float d):kp(p),ki(i),kd(d),e_pre_1(0),e_pre_2(0),target(0),actual(0)
{
   A=kp+ki+kd;
   B=-2*kd-kp;
   C=kd;
//    e=target-actual;
}
float PID_incremental::pid_control(float tar,float act)
{
   float u_increment;
   target=tar;
   actual=act;
   e=target-actual;
   u_increment=A*e+B*e_pre_1+C*e_pre_2;
   e_pre_2=e_pre_1;
   e_pre_1=e;
   return u_increment;
}

float PID_incremental::pid_control_ACC(float tar, float act, float _kp, float _ki, float _kd)
{
   float u_increment;
   float _A=_kp+_ki+_kd;
   float _B=-2*_kd-_kp;
   float _C=_kd;
   target=tar;
   actual=act;
   e=target-actual;
   u_increment=_A*e+_B*e_pre_1+_C*e_pre_2;
   e_pre_2=e_pre_1;
   e_pre_1=e;
   return u_increment;
}

void PID_incremental::pid_show()
{
    using std::cout;
    using std::endl;
    cout<<"The infomation of this incremental PID controller is as following:"<<endl;
    cout<<"     Kp="<<kp<<endl;
    cout<<"     Ki="<<ki<<endl;
    cout<<"     Kd="<<kd<<endl;
    cout<<" target="<<target<<endl;
    cout<<" actual="<<actual<<endl;
    cout<<"      e="<<e<<endl;
    cout<<"e_pre_1="<<e_pre_1<<endl;
    cout<<"e_pre_2="<<e_pre_2<<endl;
}

float P_speed(float target, float s3_speed){

    float P = 0.03333333 * 1.f ;
    float offset = 0.75f;
    float v = 0.f;

    PID_incremental pid(P,0.09,0.09);
    v=pid.pid_control(target,s3_speed);

    if(v > 1.5f){
        //v = 1.5f;
    }
    if( v <= 0.0f){
        v = 0.0f;
    }

    return v;
}

void* S3_speed_v(void* data) {

    float P = 0.031666667 * 1.f ;
    float gundata = 0.f;
    PID_incremental throttle(P,0.05,10);
    float gundata_A = 0.0f;
    float gundata_D = 0.0f;
    float target_speed_add = 0.0f;

    float _kp = 0.031666667;
    float _ki = 0.05;
    float _kd = 10;

    while(1){
        
        // pthread_mutex_lock(&mutex_s3);
        // std::cout << "target_speed = " << target_speed << std::endl;
        /*
        if(S3.speed <= target_speed * 0.1){
            _kp = 0.031666667;
            _ki = 0.075;
            _kd = 0.08;
        }
        else if(S3.speed > target_speed * 0.1 && S3.speed <= target_speed * 0.2){
            _kp = 0.031666667;
            _ki = 0.075;
            _kd = 0.08;
        }
        else if(S3.speed > target_speed * 0.2 && S3.speed <= target_speed * 0.3){
            _kp = 0.031666667;
            _ki = 0.1;
            _kd = 0.08;
        }
        else if(S3.speed > target_speed * 0.3 && S3.speed <= target_speed * 0.4){
            _kp = 0.031666667;
            _ki = 0.125;
            _kd = 0.08;
        }
        else if(S3.speed > target_speed * 0.4 && S3.speed <= target_speed * 0.5){
            _kp = 0.031666667;
            _ki = 0.15;
            _kd = 0.28;
        }
        else if(S3.speed > target_speed * 0.5 && S3.speed <= target_speed * 0.55){
            _kp = 0.031666667;
            _ki = 0.175;
            _kd = 0.28;
        }
        else if(S3.speed > target_speed * 0.55 && S3.speed <= target_speed * 0.6){
            _kp = 0.031666667;
            _ki = 0.2;
            _kd = 0.28;
        }
        else if(S3.speed > target_speed * 0.6 && S3.speed <= target_speed * 0.65){
            _kp = 0.031666667;
            _ki = 0.3;
            _kd = 0.28;
        }
        else if(S3.speed > target_speed * 0.65 && S3.speed <= target_speed * 0.7){
            _kp = 0.031666667;
            _ki = 0.35;
            _kd = 0.28;
        }
        else if(S3.speed > target_speed * 0.7 && S3.speed <= target_speed * 0.75){
            _kp = 0.031666667;
            _ki = 0.4;
            _kd = 0.28;
        }
        else if(S3.speed > target_speed * 0.75 && S3.speed <= target_speed * 0.8){
            _kp = 0.031666667;
            _ki = 0.45;
            _kd = 0.28;
        }
        else if(S3.speed > target_speed * 0.8){
            _kp = 0.031666667;
            _ki = 0.5;
            _kd = 0.28;
        }*/

        if((S3.speed - target_speed) >= target_speed * 0.7){
            _kp = 0.031666667;
            _ki = 0.1;
            _kd = 0.08;
        }
        else if((S3.speed - target_speed) >= target_speed * 0.5){
            _kp = 0.031666667;
            _ki = 0.125;
            _kd = 0.12;
        }
        else if((S3.speed - target_speed) >= target_speed * 0.3){
            _kp = 0.031666667;
            _ki = 0.15;
            _kd = 0.15;
        }
        else{
            _kp = 0.031666667;
            _ki = 0.175;
            _kd = 0.2;
        }

        gundata = throttle.pid_control_ACC(target_speed, S3.speed, _kp, _ki, _kd);
        if( gundata <= 0.0f){
            gundata = 0.75f;
        }
        else if( gundata >= 2.8f){
            gundata = 2.8f;
        }
        
        if(target_speed == 0){
            gundata = 0.0f;
        }

        canbus_ctrl_pedal(gundata);
        // cout << "gundata " << gundata << "," << "S3.speed" << S3.speed << endl;
        usleep(2000);
    } 
}

void* S3_dec(void* data) {

    float break_value = 0.f;
    float break_add_value = 0.0f;
    PID_incremental brakes_f(0.05,0.075,0.2); //0.02,0.075,0.1

    float _kp_c1 = 0.03;
    float _ki_c1 = 0.01;
    float _kd_c1 = 0.2;

    float _kp_c2 = 0.05;
    float _ki_c2 = 0.1;
    float _kd_c2 = 0.2;
    while(1){
        if(stop_flag == 1){
            _ki_c1 = _ki_c1 + 0.001;
            // if(_ki_c1>0.1){
            //     _ki_c1 = 0.1;
            // }
            break_value = brakes_f.pid_control_ACC(Targetdistance, S3.speed, _kp_c1, _ki_c1, _kd_c1);
            if(break_value < 0.f){
                break_value = break_value * (-1);
            }
            else{
                break_value = 0.f;
            }
            deceleration = break_value;
            _ki_c2 = _ki_c1;
        }
        else if(stop_flag == 2){
            _ki_c2 = _ki_c2 + 0.0001;
            if(_ki_c2 > 0.45){
                _ki_c2 = 0.45;
            }
            break_value = brakes_f.pid_control_ACC(Targetdistance, S3.speed, _kp_c2, _ki_c2, _kd_c2);
            if(break_value < 0.f){
                break_value = break_value * (-1);
            }
            else{
                break_value = 0.f;
            }
            deceleration = break_value;
            
            _ki_c1 = 0;
        }
        else if(stop_flag == 0){
            deceleration -= 0.001f;
            if(deceleration < 0.f){
                deceleration = 0.0f;
            }
            _ki_c1 = 0;
            _ki_c2 =0;
        }
        // cout << "S3 speed " << S3.speed << endl;
        // cout << "deceleration " << break_value << endl;
        usleep(3000);
    } 
}

void Dynamic_System(){
	pthread_t t_S3_v; // 宣告 pthread 變數
    pthread_t t_S3_dec; // 宣告 pthread 變數
    pthread_t t_sound; // 宣告 pthread 變數
    pthread_create(&t_S3_v, NULL, S3_speed_v, NULL);
    pthread_create(&t_S3_dec, NULL, S3_dec, NULL);
}

#endif