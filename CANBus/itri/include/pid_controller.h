#pragma once
# include <iostream>

//增量式PID
class PID_incremental
{
private:
    float target;
    float actual;
    float e;
    float A;
    float B;
    float C;
    float kp;
    float ki;
    float kd;
public:
    
    PID_incremental();
    PID_incremental(float p,float i,float d);
    float pid_control(float tar, float act);
    float pid_control_ACC(float tar, float act, float _kp, float _ki, float _kd);
    float e_pre_1;
    float e_pre_2;
    void pid_show();
};

float P_speed(float target, float s3_speed);
void* S3_speed_v(void* data);
void* S3_dec(void* data);