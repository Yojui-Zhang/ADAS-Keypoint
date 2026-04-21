#ifndef CANBUS_RECV_H
#define CANBUS_RECV_H

#include <cstdint>
#include <string>
#include <vector>
#include <net/if.h>
#include <linux/can.h>
#include <linux/can/raw.h>


enum WHEEL_MODE
{
	WM_SET_ANGLE,
	WM_SET_ANGULAR_VELO
};


class CANBUS
{
public:
	int createSocket();
	void setInterface(std::string);
	void linkToAddr();
	void setFilter();
	void setSocket();
	struct sockaddr_can &getSocketAddr();
	int bindSocket();

	struct sockaddr_can addr;
	struct canfd_frame ECUMsg1;	// ID: 200
	struct canfd_frame ECUMsg2;	// ID: 201
	struct canfd_frame ABS1;	// ID: 302
	struct canfd_frame ABS4;	// ID: 303
	struct canfd_frame EPAS1;	// ID: 307
	struct canfd_frame EPAS2;	// ID: 308


	struct canfd_frame AutoGear;	// ID: 250
	struct canfd_frame AutoPedal;
private:
	int s;	// socket
	struct ifreq ifr;
	struct can_filter canFilter[3];
	const int canfd_on = 1;
};

enum COORDINATE
{
	CW,		// 順時針(+)
	CCW		// 逆時針(+)
};

typedef struct CAR
{
	/* vehicle parameter */
	double frontBumper_L;
	double backBumper_L;
	double wheelbase;
	double track_F;
	double track_B;
	double length;
	double width;

	/* receive from CAN bus. */
	double speed;
	double speedOri;
	double theta;
	double phi;
	double steer;
	double steeringTorque;
	double yaw;
	double latAccel;
	double longAccel;
	int throttle;
	unsigned char gear;
	bool SAS_CAL;
	unsigned char turningSignal;
	int motorTorque;
	uint64_t can_last_rx_sync_ns;
	uint64_t can_powertrain_rx_sync_ns;
	uint64_t can_speed_rx_sync_ns;
	uint64_t can_yaw_rx_sync_ns;
	uint64_t can_steer_rx_sync_ns;
	uint64_t can_steering_torque_rx_sync_ns;
	uint64_t can_turn_signal_rx_sync_ns;

	/*receive from Radar CAN bus. */
	int packageID;
    int endPoint;
    int reserved;
    int RadarX;
    int RadarY;
    int RadarP;
    int RadarV;
	double RadarR;
    double RadarV_f;


	/* calculata from velocity, steer angle or tire angle... */
	double thetaBefore;
	double meterage;
	double beforeSpeed;
	double accel;
	double tireAngle;
	double turningRadius[10];


	/* vehicle parameter setting */
	bool tireCoordSys;	// tire coordinate system
	double tire_max;
	double wheel_max;

	void cal_radius();
	void cal_radius(double _tireAngle_, double R[10]);


}CAR;

struct S {
	unsigned speed : 12;			// ID: 302		Data: 63~52
	unsigned : 0;
	unsigned steer : 16;			// ID: 307 		Data: 63~48
	unsigned angleTarget : 16;		// ID: 201 		Data: 55~40
	unsigned steeringTorque : 8;	// ID: 308 		Data: 47~40
	unsigned EPS_Sta_Available : 2;	// ID: 307 		Data: 23~22
	unsigned V_Rq_EPS_Ctrl : 1;		// ID: 201 		Data: 57
	unsigned Rq_EPS_Ctrl : 1;		// ID: 201 		Data: 56
	unsigned SAS_CAL : 1;			// ID: 307 		Data: 33
	unsigned yaw : 16;				// ID: 303		Data: 55~40
	unsigned latAccel : 8;			// ID: 303		Data: 39~32
	unsigned longAccel : 8;			// ID: 303		Data: 31~24
	unsigned throttle : 8;			// ID: 300		Data: 15~8
	unsigned gear : 3;
	unsigned turningSignal : 2;
	unsigned motorTorque : 8;		// ID: 300		Data: 47~40

	unsigned RADAR_packageID : 6;	// ID: 0C1		Data: 63~58
    unsigned RADAR_endPoint : 2;	// ID: 0C1		Data: 57~56
    unsigned RADAR_reserved : 4;	// ID: 0C1		Data: 55~52
    unsigned RADAR_X : 10;			// ID: 0C1		Data: 51~42
    unsigned RADAR_Y : 13;			// ID: 0C1		Data: 41~29
    unsigned RADAR_P : 5;			// ID: 0C1		Data: 28~24
    unsigned RADAR_R : 12;			// ID: 0C1		Data: 23~12
    unsigned RADAR_V : 12;			// ID: 0C1		Data: 11~0
};

enum GEAR
{
	PARK = 0, NEUTRAL = 4, DRIVE, FAILURE, REVERSE
};
enum TURN_SIGNAL
{
	OFF, LEFT, RIGHT, INVALID
};

extern std::vector<int> radarX;
extern std::vector<int> radarY;
extern std::vector<int> radarP;
extern std::vector<double> radarR;
extern std::vector<double> radarV;

extern void canbus_recv(CAR &car);
extern void canbus_ctrl_steer(int SW);
extern void canbus_set_steering_tx_enabled(int enabled);
extern void canbus_ctrl_pedal(double pedalDst);
extern void canbus_ctrl_gear(int gearDst);
extern void canbus_ctrl_dec(int SW);
extern void canbus_stop_dec();
extern void wait_pthread();
#endif
