#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <signal.h>
#include <ctype.h>
#include <libgen.h>
#include <time.h>
#include <errno.h>
#include <pthread.h>
#include <iostream>
#include <cstdio>
#include <cmath>

#include <sys/time.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <sys/uio.h>

#include "../include/canbus_recv.h"
#include "../include/terminal.h"
#include "../include/lib.h"
#include "time_sync.h"

#define MAXSOCK 16    /* max. number of CAN interfaces given on the cmdline */
#define MAXIFNAMES 30 /* size of receive name index to omit ioctls */
#define ANYDEV "any"  /* name of interface to receive from any CAN interface */
#define ANL "\r\n"    /* newline in ASC mode */

// #define DEBUG

#define DEG_TO_RAD(theta) ((theta)*0.01745329251994329576923690768489)
#define RAD_TO_DEG(rad) ((rad)*57.29577951308232087679815481410)
#define SQUARE(x) (double(x)*double(x))

using namespace std;

int turn_num = 0;
float car_turn_angle = 0;
float lidar_distance = 0;
int number_1;

volatile int SteerCtrlSwitch = 0;
volatile int steerCtrlMode = WM_SET_ANGLE;
volatile double targetAngle;

volatile double deceleration = 0.0;	// m/s

// volatile int old_packageID = 0;
// volatile int endpoint_state = 1;
// std::vector<int> radarX;
// std::vector<int> radarY;
// std::vector<int> radarP;
// std::vector<double> radarR;
// std::vector<double> radarV;

CAR *ptr_car;

pthread_t canWheel;
pthread_t canDec;
pthread_t canRecvSend;
pthread_t can1_pth;

class TIME
{
public:
	void getStartTime()
	{
		gettimeofday(&start, NULL);
	}

	void getEndTime()
	{
		gettimeofday(&end, NULL);
	}
	double diffTime()
	{
		return double(((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000000.0);
	}

private:
	struct timeval start, end;

}t1,t2,t3,t4,t5,t6,t_meterage, t_yaw, t_wheelRecv;


int CANBUS::createSocket()
{
	s = socket(PF_CAN, SOCK_RAW, CAN_RAW);
	return s;	// Open a socket for CAN bus.
}
void CANBUS::setInterface(std::string interface)
{
	const char *__interface__ = interface.data();
	memset(&ifr.ifr_name, 0, sizeof(ifr.ifr_name));
	strncpy(ifr.ifr_name, __interface__, 4);
}
void CANBUS::linkToAddr()
{
	addr.can_family = AF_CAN;
	if (strcmp(ANYDEV, ifr.ifr_name)) {
		if (ioctl(s, SIOCGIFINDEX, &ifr) < 0) {	// Get a network interface address.
			perror("SIOCGIFINDEX");
			exit(1);
		}
		addr.can_ifindex = ifr.ifr_ifindex;
	}
	else
		addr.can_ifindex = 0; /* any can interface */

	printf("addr.can_ifindex: %d ifr.ifr_ifindex: %d\n", addr.can_ifindex, ifr.ifr_ifindex);
}
void CANBUS::setFilter()
{
	canFilter[0].can_mask = 0x30f;
	canFilter[0].can_id = 0x302;
	canFilter[0].can_mask &= ~CAN_ERR_FLAG;

	canFilter[2].can_mask = 0x30f;
	canFilter[2].can_id = 0x303;
	canFilter[2].can_mask &= ~CAN_ERR_FLAG;

	canFilter[1].can_mask = 0x30f;
	canFilter[1].can_id = 0x307;
	canFilter[1].can_mask &= ~CAN_ERR_FLAG;

	canFilter[2].can_mask = 0x30f;
	canFilter[2].can_id = 0x308;
	canFilter[2].can_mask &= ~CAN_ERR_FLAG;

	canFilter[3].can_mask = 0x30f;
	canFilter[3].can_id = 0x309;
	canFilter[3].can_mask &= ~CAN_ERR_FLAG;

	// canFilter[4].can_mask = 0xCf;
    // canFilter[4].can_id = 0xC1;
    // canFilter[4].can_mask &= ~CAN_ERR_FLAG;
}
void CANBUS::setSocket()
{
	// setsockopt(s, SOL_CAN_RAW, CAN_RAW_FILTER, canFilter, sizeof(struct can_filter) * 4);

	/* try to switch the socket into CAN FD mode */
	setsockopt(s, SOL_CAN_RAW, CAN_RAW_FD_FRAMES, &canfd_on, sizeof(canfd_on));	// 在CAN_RAW套接字中啟用CAN FD支持
}
struct sockaddr_can & CANBUS::getSocketAddr()
{
	return CANBUS::addr;
}
int CANBUS::bindSocket()
{
	return (bind(s, (struct sockaddr *)&addr, sizeof(addr)));
}
struct S canData;
int s;
int ret;
int nbytes, maxdlen;
unsigned char view = 0;
struct iovec iov;
struct msghdr msg;
struct canfd_frame canFrame;	// CAN ID and data.
struct cmsghdr *cmsg;
struct timeval tv, last_tv;
char ctrlmsg[CMSG_SPACE(sizeof(struct timeval)) + CMSG_SPACE(sizeof(__u32))];
fd_set rdfs;

struct timeval timeout, timeout_config = { 0, 0 }, *timeout_current = NULL;
CANBUS ECU;

//=======can1===============
int s_can1;
CANBUS ECU_can1;
//=======can1===============


const char col_off[] = ATTRESET;

static char *cmdlinename[MAXSOCK];
static __u32 dropcnt[MAXSOCK];
static __u32 last_dropcnt[MAXSOCK];
static char devname[MAXIFNAMES][IFNAMSIZ + 1];
static int  dindex[MAXIFNAMES];
static int  max_devname_len; /* to prevent frazzled device name output */
const int canfd_on = 1;


extern int optind, opterr, optopt;

static volatile int running = 1;

// static int check_SAS_CAL();
// static inline void send_V_Rq_EPS_Ctrl();
// static inline void send_Rq_EPS_Ctrl();
// static inline void angleDecToCanbus(double angleIn, char canbusOut[2]);
// static inline void send_target_angle(double angle);

void CAR::cal_radius()
{
	double phiRad = DEG_TO_RAD(tireAngle);
	double L = wheelbase;	// 輪距
	double K_front = frontBumper_L;
	double K_back = backBumper_L;
	double T_front = track_F;
	double T_back = track_B;
	double(&R)[10] = turningRadius;
	if(tireCoordSys == CW)
	{
		R[0] = L / tan(phiRad);
		R[1] = sqrt(SQUARE(R[0]) + SQUARE(L));
		R[2] = R[0] + T_back / 2;
		R[3] = R[0] - T_back / 2;
		R[4] = sqrt(SQUARE(R[0] + T_front / 2) + SQUARE(L));
		R[5] = sqrt(SQUARE(R[0] - T_front / 2) + SQUARE(L));
		R[6] = sqrt(SQUARE(R[0] + width / 2) + SQUARE(K_back));
		R[7] = sqrt(SQUARE(R[0] - width / 2) + SQUARE(K_back));
		R[8] = sqrt(SQUARE(R[0] + width / 2) + SQUARE(L + K_front));
		R[9] = sqrt(SQUARE(R[0] - width / 2) + SQUARE(L + K_front));
	}
	else if(tireCoordSys == CCW)
	{
		R[0] = L / tan(phiRad);
		R[1] = sqrt(SQUARE(R[0]) + SQUARE(L));
		R[2] = R[0] - T_back / 2;
		R[3] = R[0] + T_back / 2;
		R[4] = sqrt(SQUARE(R[0] - T_front / 2) + SQUARE(L));
		R[5] = sqrt(SQUARE(R[0] + T_front / 2) + SQUARE(L));
		R[6] = sqrt(SQUARE(R[0] - width / 2) + SQUARE(K_back));
		R[7] = sqrt(SQUARE(R[0] + width / 2) + SQUARE(K_back));
		R[8] = sqrt(SQUARE(R[0] - width / 2) + SQUARE(L + K_front));
		R[9] = sqrt(SQUARE(R[0] + width / 2) + SQUARE(L + K_front));
	}

	for(int i=0;i<10;i++)
	{
		R[i] = fabs(R[i]);
	}
}

void CAR::cal_radius(double _tireAngle_, double R[10])
{
	double phiRad = DEG_TO_RAD(_tireAngle_);
	double L = wheelbase;	// 輪距
	double K_front = frontBumper_L;
	double K_back = backBumper_L;
	double T_front = track_F;
	double T_back = track_B;
	if (tireCoordSys == CW)
	{
		R[0] = L / tan(phiRad);
		R[1] = sqrt(SQUARE(R[0]) + SQUARE(L));
		R[2] = R[0] + T_back / 2;
		R[3] = R[0] - T_back / 2;
		R[4] = sqrt(SQUARE(R[0] + T_front / 2) + SQUARE(L));
		R[5] = sqrt(SQUARE(R[0] - T_front / 2) + SQUARE(L));
		R[6] = sqrt(SQUARE(R[0] + width / 2) + SQUARE(K_back));
		R[7] = sqrt(SQUARE(R[0] - width / 2) + SQUARE(K_back));
		R[8] = sqrt(SQUARE(R[0] + width / 2) + SQUARE(L + K_front));
		R[9] = sqrt(SQUARE(R[0] - width / 2) + SQUARE(L + K_front));
	}
	else if (tireCoordSys == CCW)
	{
		R[0] = L / tan(phiRad);
		R[1] = sqrt(SQUARE(R[0]) + SQUARE(L));
		R[2] = R[0] - T_back / 2;
		R[3] = R[0] + T_back / 2;
		R[4] = sqrt(SQUARE(R[0] - T_front / 2) + SQUARE(L));
		R[5] = sqrt(SQUARE(R[0] + T_front / 2) + SQUARE(L));
		R[6] = sqrt(SQUARE(R[0] - width / 2) + SQUARE(K_back));
		R[7] = sqrt(SQUARE(R[0] + width / 2) + SQUARE(K_back));
		R[8] = sqrt(SQUARE(R[0] - width / 2) + SQUARE(L + K_front));
		R[9] = sqrt(SQUARE(R[0] + width / 2) + SQUARE(L + K_front));
	}

	for (int i = 0; i<10; i++)
	{
		R[i] = fabs(R[i]);
	}
}

void sigterm(int signo)
{
	running = 0;
}

int idx2dindex(int ifidx, int socket) {

	int i;
	struct ifreq ifr;

	for (i = 0; i < MAXIFNAMES; i++) {
		if (dindex[i] == ifidx)
			return i;
	}

	/* create new interface index cache entry */

	/* remove index cache zombies first */
	for (i = 0; i < MAXIFNAMES; i++) {
		if (dindex[i]) {
			ifr.ifr_ifindex = dindex[i];
			if (ioctl(socket, SIOCGIFNAME, &ifr) < 0)
				dindex[i] = 0;
		}
	}

	for (i = 0; i < MAXIFNAMES; i++)
		if (!dindex[i]) /* free entry */
			break;

	if (i == MAXIFNAMES) {
		fprintf(stderr, "Interface index cache only supports %d interfaces.\n",
			MAXIFNAMES);
		exit(1);
	}

	dindex[i] = ifidx;

	ifr.ifr_ifindex = ifidx;
	if (ioctl(socket, SIOCGIFNAME, &ifr) < 0)
		perror("SIOCGIFNAME");

	if (max_devname_len < strlen(ifr.ifr_name))
		max_devname_len = strlen(ifr.ifr_name);

	strcpy(devname[i], ifr.ifr_name);

#ifdef DEBUG
	printf("new index %d (%s)\n", i, devname[i]);
#endif

	return i;
}


//=======can1===============

void* pth_can1(void *data)
{
	CAR *ptr_car = (CAR *)data;
	int i = 0;

	while (running)
	{
	receive:
		FD_ZERO(&rdfs);		   // 將 set 清空使集合中不含任何 fd
		FD_SET(s_can1, &rdfs); // 將 fd 加入 set 集合中，改為 s_can1

		if ((ret = select(s_can1 + 1, &rdfs, NULL, NULL, timeout_current)) <= 0)
		{
			running = 0;
			continue;
		}

		if (FD_ISSET(s_can1, &rdfs))
		{

			int idx;

			/* these settings may be modified by recvmsg() */
			iov.iov_len = sizeof(canFrame);
			msg.msg_namelen = sizeof(ECU_can1.getSocketAddr()); // 使用 ECU_can1 的地址
			msg.msg_controllen = sizeof(ctrlmsg);
			msg.msg_flags = 0;

			nbytes = recvmsg(s_can1, &msg, 0);								// 改為 s_can1 接收 CAN1 訊息
			idx = idx2dindex(ECU_can1.getSocketAddr().can_ifindex, s_can1); // 使用 ECU_can1

			if ((size_t)nbytes == CAN_MTU)
				maxdlen = CAN_MAX_DLEN;
			else if ((size_t)nbytes == CANFD_MTU)
				maxdlen = CANFD_MAX_DLEN;
			else
			{
				fprintf(stderr, "read: incomplete CAN frame\n");
				pthread_exit(NULL); // 離開子執行緒
			}

			for (cmsg = CMSG_FIRSTHDR(&msg);
				 cmsg && (cmsg->cmsg_level == SOL_SOCKET);
				 cmsg = CMSG_NXTHDR(&msg, cmsg))
			{
				if (cmsg->cmsg_type == SO_TIMESTAMP)
					memcpy(&tv, CMSG_DATA(cmsg), sizeof(tv));
				else if (cmsg->cmsg_type == SO_RXQ_OVFL)
					memcpy(&dropcnt[0], CMSG_DATA(cmsg), sizeof(__u32));
			}

			/* once we detected a EFF frame indent SFF frames accordingly */
			switch (canFrame.can_id)
			{
			case 0x301:
				number_1 = canFrame.data[0];
				cout << "can0 301 : " << number_1 << endl;
				break;

			case 0x0C2:
				lidar_distance = (canFrame.data[0] << 8) + canFrame.data[1];
				break;
			}
			
		}
	}
	pthread_exit(NULL); // 離開子執行緒
}


//=======can1===============

void* pth_canRecv(void *data) {
	CAR *ptr_car = (CAR*)data;
	int i = 0;

	bool isValid_SAS_CAL = 0;
	bool isValid_VehSpeed = 0;
	bool isValid_SteeringTorque = 0;
	bool isInCtrlWheelMode_Rq = 0;
	
	double errorSum = 0.0;
	double error = 0.0;
	t_meterage.getEndTime();
	t_meterage.getStartTime();
	t_yaw.getEndTime();
	t_yaw.getStartTime();
	t_wheelRecv.getStartTime();
	t_wheelRecv.getEndTime();

	//-----------------
	double theta_delta = 0.0;
	//----------------
	double &speed = ptr_car->speed;
	double &speedOri = ptr_car->speedOri;
	double &beforeSpeed = ptr_car->beforeSpeed;
	double &accel = ptr_car->accel;
	double &steer = ptr_car->steer;
	double &meterage = ptr_car->meterage;
	double &yaw = ptr_car->yaw;
	double &theta = ptr_car->theta;
	double &thetaBefore = ptr_car->thetaBefore;
	double &latAccel = ptr_car->latAccel;
	double &longAccel = ptr_car->longAccel;
	double &tireAngle = ptr_car->tireAngle;
	unsigned char &gear = ptr_car->gear;
	double &steeringTorque = ptr_car->steeringTorque;
	bool &SAS_CAL = ptr_car->SAS_CAL;
	int &throttle = ptr_car->throttle;
	unsigned char &turningSignal = ptr_car->turningSignal;
	int &motorTorque = ptr_car->motorTorque;
	double &tire_max = ptr_car->tire_max;
	double &wheel_max = ptr_car->wheel_max;

	// int &packageID = ptr_car->packageID;
    // int &endPoint = ptr_car->endPoint;
    // int &reserved = ptr_car->reserved;
    // int &RadarX = ptr_car->RadarX;
    // int &RadarY = ptr_car->RadarY;
    // int &RadarP = ptr_car->RadarP;
    // int &RadarV = ptr_car->RadarV;
	// double &RadarR = ptr_car->RadarR;
    // double &RadarV_f = ptr_car->RadarV_f;

	//----------------


	while (running)
	{
	receive:
		FD_ZERO(&rdfs);		// 將set清空使集合中不含任何fd
		FD_SET(s, &rdfs);	// 將fd加入set集合中
		if ((ret = select(s + 1, &rdfs, NULL, NULL, timeout_current)) <= 0) {
			running = 0;
			continue;
		}

		{
			if (FD_ISSET(s, &rdfs))
			{

				int idx;

				/* these settings may be modified by recvmsg() */
				iov.iov_len = sizeof(canFrame);
				msg.msg_namelen = sizeof(ECU.addr);
				msg.msg_controllen = sizeof(ctrlmsg);
				msg.msg_flags = 0;

					nbytes = recvmsg(s, &msg, 0);
					idx = idx2dindex(ECU.addr.can_ifindex, s);


				if ((size_t)nbytes == CAN_MTU)
					maxdlen = CAN_MAX_DLEN;
				else if ((size_t)nbytes == CANFD_MTU)
					maxdlen = CANFD_MAX_DLEN;
				else {
					fprintf(stderr, "read: incomplete CAN frame\n");
					pthread_exit(NULL); // 離開子執行緒
				}


					for (cmsg = CMSG_FIRSTHDR(&msg);
					cmsg && (cmsg->cmsg_level == SOL_SOCKET);
						cmsg = CMSG_NXTHDR(&msg, cmsg)) {
						if (cmsg->cmsg_type == SO_TIMESTAMP)
							memcpy(&tv, CMSG_DATA(cmsg), sizeof(tv));
						else if (cmsg->cmsg_type == SO_RXQ_OVFL)
							memcpy(&dropcnt[0], CMSG_DATA(cmsg), sizeof(__u32));
					}

					const uint64_t rx_sync_ns = TimeSyncNowNs();

					/* once we detected a EFF frame indent SFF frames accordingly */
					switch (canFrame.can_id)
					{
					case 0x300:
						ptr_car->can_last_rx_sync_ns = rx_sync_ns;
						ptr_car->can_powertrain_rx_sync_ns = rx_sync_ns;
						canData.gear = (canFrame.data[5]&0xe0) >> 5;
						gear = canData.gear;
						motorTorque = int(canFrame.data[2])*2-160;
						throttle = int(canFrame.data[6]);
						break;

					case 0x302:
						ptr_car->can_last_rx_sync_ns = rx_sync_ns;
						ptr_car->can_speed_rx_sync_ns = rx_sync_ns;
						t_meterage.getEndTime();
						ECU.ABS1=canFrame;

					{
						canData.speed = (canFrame.data[0] << 4) | (canFrame.data[1] >> 4);
						// speed = (double)(canData.speed) / 8.0;
						speedOri = (double)(canData.speed) / 8.0;
						double v_hat = (double)(canData.speed) / 8.0;
						double v1 = speed;
						double v2 = v1 + (longAccel * t_meterage.diffTime())*3.6;
						static double P=1;
						static double Q=0.1;
						static double R=0.1;
						P=P+Q;
						static double K=0;
						K = P/(P+R);
						P = (1 - K) * P;
						speed = v2 + K*(v_hat - v2);

						
						double _meterage_ = (beforeSpeed + speed)*(t_meterage.diffTime()) / (2 * 3.6)*0.95;
						meterage = meterage + _meterage_;
						accel = ((speed - beforeSpeed)/3.6)/(t_meterage.diffTime());
						beforeSpeed = speed;

					}
					t_meterage.getStartTime();
					// gettimeofday(&tv_Meterage[0], NULL);
					break;

					case 0x303:
						ptr_car->can_last_rx_sync_ns = rx_sync_ns;
						ptr_car->can_yaw_rx_sync_ns = rx_sync_ns;
						t_yaw.getEndTime();
						ECU.ABS4 = canFrame;

					canData.yaw = (ECU.ABS4.data[1]<<8) | (ECU.ABS4.data[2]);
					canData.latAccel = ECU.ABS4.data[3];
					canData.longAccel = ECU.ABS4.data[4];

					yaw = (canData.yaw-1000)*0.1;
					latAccel = (canData.latAccel - 127)*0.01*9.8;
					longAccel = (canData.longAccel - 127)*0.01*9.8;
					if(fabs(longAccel)<=0.099) longAccel = 0.0;


					theta = thetaBefore + yaw*(t_yaw.diffTime());
					thetaBefore = theta;

					t_yaw.getStartTime();
					break;

					case 0x307:
						ptr_car->can_last_rx_sync_ns = rx_sync_ns;
						ptr_car->can_steer_rx_sync_ns = rx_sync_ns;
						t_wheelRecv.getEndTime();
						ECU.EPAS1 = canFrame;

					SAS_CAL = (ECU.EPAS1.data[3] & 0x02)>>1;
					canData.EPS_Sta_Available = ECU.EPAS1.data[5] & 0xc0;
					canData.steer = (canFrame.data[0] << 8) | (canFrame.data[1]);


					steer = (double)(canData.steer - 9000)*0.1;

					if (wheel_max < 1e-6) {
						wheel_max = 540.0;   // 例：方向盤最大角度(度)，用你車實測值替換
					}
					if (tire_max < 1e-6) {
						tire_max = 35.0;     // 例：前輪最大轉角(度)，用你車實測值替換
					}

					tireAngle = steer * tire_max/wheel_max;
					ptr_car->cal_radius();

					t_wheelRecv.getStartTime();
					break;

					case 0x308:
						ptr_car->can_last_rx_sync_ns = rx_sync_ns;
						ptr_car->can_steering_torque_rx_sync_ns = rx_sync_ns;
						ECU.EPAS2 = canFrame;

					canData.steeringTorque = ECU.EPAS2.data[2];
					steeringTorque = (double)(canData.steeringTorque - 127)*0.1794;
					break;

					case 0x309:
						ptr_car->can_last_rx_sync_ns = rx_sync_ns;
						ptr_car->can_turn_signal_rx_sync_ns = rx_sync_ns;
						canData.turningSignal = (canFrame.data[3]&0x30)>>4;
						turningSignal = canData.turningSignal;
						break;

				//case 0xC1:
				//	turn_num = canFrame.data[0];  //
				//	car_turn_angle = float((canFrame.data[1] - 180));
				//	break;
				


				// 	// packageID
                //     canData.RADAR_packageID = (canFrame.data[0] & 0xfc) >> 2;
                //     packageID = (int)canData.RADAR_packageID;
				// 	// endPoint
                //     canData.RADAR_endPoint = canFrame.data[0] & 0x03;
                //     endPoint = (int)canData.RADAR_endPoint;
				// 	// reserved
                //     canData.RADAR_reserved = (canFrame.data[1] & 0xf0) >> 4;
                //     reserved = (int)canData.RADAR_reserved;
				// 	// X
                //     if((int)(canFrame.data[1] & 0x08) == 8)
                //     {
				// 		// RadarX = -((int)((canFrame.data[1] & 0x0f) << 6) + (int)((canFrame.data[2] & 0xfc) >> 2));
                //         RadarX = ~((int)((canFrame.data[1] & 0x0f) << 6) + (int)((canFrame.data[2] & 0xfc) >> 2) - 1);
                //         RadarX = -(RadarX & 1023);
                //     }
                //     else if((int)(canFrame.data[1] & 0x08) == 0)
                //     {
                //         canData.RADAR_X = ((canFrame.data[1] & 0x07) << 6)|((canFrame.data[2] & 0xfc) >> 2);
                //         RadarX = (int)canData.RADAR_X;
                //     }       
				// 	// Y
                //     if((int)(canFrame.data[2] & 0x02) == 2)
                //     {
				// 		// RadarY = -((int)((canFrame.data[2] & 0x03) << 11) + (int)((canFrame.data[3] & 0xff) << 3) + (int)((canFrame.data[4] & 0xe0) >> 5));
                //         RadarY = ~((int)((canFrame.data[2] & 0x03) << 11) + (int)((canFrame.data[3] & 0xff) << 3) + (int)((canFrame.data[4] & 0xe0) >> 5) - 1);
                //         RadarY = -(RadarY & 8191);
                //     }
                //     else if((int)(canFrame.data[1] & 0x02) == 0)
                //     {
                //         canData.RADAR_Y = ((canFrame.data[2] & 0x01) << 11)|((canFrame.data[3] & 0xff) << 3)|((canFrame.data[4] & 0xe0) >> 5);
                //         RadarY = (int)canData.RADAR_Y;
                //     }
				// 	// P
                //     canData.RADAR_P = canFrame.data[4] & 0x1f;
                //     RadarP = (int)canData.RADAR_P * 2;
				// 	// R
                //     canData.RADAR_R = ((canFrame.data[5] & 0xff) << 4)|((canFrame.data[6] & 0xf0) >> 4);
                //     RadarR = (double)canData.RADAR_R * 0.1;
				// 	// V
                //     if((int)(canFrame.data[6] & 0x08) == 8)
                //     {
				// 		// RadarV = -((int)((canFrame.data[6] & 0x0f) << 8) + (int)(canFrame.data[7] & 0xff));
                //         RadarV = ~((int)((canFrame.data[6] & 0x0f) << 8) + (int)(canFrame.data[7] & 0xff) - 1);
                //         RadarV = -(RadarV & 4095);
                //         RadarV_f = RadarV * 0.1;
                //     }
                //     else if((int)(canFrame.data[6] & 0x08) == 0)
                //     {
                //         RadarV = (int)((canFrame.data[6] & 0x07) << 8) + (int)(canFrame.data[7] & 0xff);
                //         RadarV_f = RadarV * 0.1;
                //     }
                //     break;
				}

				switch(canData.gear)
				{
					case PARK: gear='P'; break;
					case NEUTRAL: gear='N'; break;
					case DRIVE: gear='D'; break;
					case REVERSE: gear='R'; break;
					default: gear='F'; break;
				}

				switch(canData.turningSignal)
				{
					case OFF: turningSignal='O'; break;
					case LEFT: turningSignal='L'; break;
					case RIGHT: turningSignal='R'; break;
					default: turningSignal='F'; break;
				}
				// float theta = 3.5 * (float)M_PI / 180;
				// float rotation_x =  cos(theta) * RadarX - sin(theta) * RadarY;
				// float rotation_y =  sin(theta) * RadarX + cos(theta) * RadarY;

				// if(old_packageID != packageID)
        		// {
				// 	endpoint_state = 1;
				// 	old_packageID = packageID;
				// 	radarX.clear();
				// 	radarY.clear();
				// 	radarP.clear();
				// 	radarR.clear();
				// 	radarV.clear();
				// }
				// if(endpoint_state == 1)
				// {
				// 	// if(rotation_x >= -18.8 && rotation_x <= 18.8)
				// 	// && RadarP >= 24
				// 	if(RadarX >= -28.2 && RadarX <= 28.2)
				// 	{
				// 		radarX.push_back(rotation_x);
				// 		radarY.push_back(rotation_y);
				// 		radarP.push_back(RadarP);
				// 		radarR.push_back(RadarR);
				// 		radarV.push_back(RadarV_f);
				// 	}
				// 	if(endPoint == 1 || endPoint == 2)
				// 	{
				// 		endpoint_state = 0;
				// 	}
				// }
			}
		}
	}
	pthread_exit(NULL); // 離開子執行緒
}





/* = = = = = = = = = = = = = = = = = = = = = 剎車控制 = = = = = = = = = = = = = = = = = = = = = */
volatile unsigned char DecCtrlSwitch = 0;
void *pth_canDec(void *data)
{
	TIME t_Dec;
	auto &ADAS_DecReq_A = DecCtrlSwitch;
	unsigned char ADAS_DecReq = 0;


	t_Dec.getStartTime();
	t_Dec.getEndTime();
	while(ADAS_DecReq_A)
	{
		t_Dec.getEndTime();
		if((t_Dec.diffTime())>0.01)
		{
			t_Dec.getStartTime();
			ADAS_DecReq = (unsigned char)(deceleration*25.4);
			
			ECU.ECUMsg1.can_id = 0x200;
			ECU.ECUMsg1.data[0] = 0x00;
			ECU.ECUMsg1.data[1] = ADAS_DecReq_A<<1;
			ECU.ECUMsg1.data[2] = ADAS_DecReq;
			ECU.ECUMsg1.data[3] = 0x00;
			ECU.ECUMsg1.data[4] = 0x00;
			ECU.ECUMsg1.data[5] = 0x00;
			ECU.ECUMsg1.data[6] = 0x00;
			ECU.ECUMsg1.data[7] = 0x00;
			ECU.ECUMsg1.len = 8;

			ret = write(s, &(ECU.ECUMsg1), 16);
			if (ret > 0) {
				TimeSyncMarkCanBrakeTxNs(TimeSyncNowNs());
			}
		}
	}
	
	deceleration = 0.0;
	ADAS_DecReq_A = 0;
	printf("Stop brake.\n");
	pthread_exit(NULL);
}

void canbus_ctrl_dec(int SW)
{
	switch(SW)
	{
	case 0:
		
		DecCtrlSwitch = 0;
		deceleration = 0.0;
		pthread_join(canDec, NULL);
		std::cout<<"Successful. brake module has been stopped."<< std::endl;
		break;

	case 1:
		// canbus_ctrl_pedal(0.75);
		if(DecCtrlSwitch == 0)
		{
			DecCtrlSwitch = 1;
			std::cout << "Boot brake module." << std::endl;
			pthread_create(&canDec, NULL, pth_canDec, (void *)ptr_car);
		}
		else if(DecCtrlSwitch == 1)
		{
			std::cout << "Brake module has been booted" << std::endl;
		}
		break;

	}
	
}

/* ========================================== */


/* = = = = = = = = = = = = = = = = = = = = = 檔位控制 = = = = = = = = = = = = = = = = = = = = = */
void canbus_ctrl_gear(int gearDst)
{
	volatile auto &gear = ptr_car->gear;
	
	auto &speed = ptr_car->speed;
    int nowGear = gearDst;


	ECU.AutoGear.can_id = 0x250;
	ECU.AutoGear.data[0] = 0x00;
	ECU.AutoGear.data[2] = 0x00;
	ECU.AutoGear.data[3] = 0x00;
	ECU.AutoGear.data[4] = 0x00;
	ECU.AutoGear.data[5] = 0x00;
	ECU.AutoGear.data[6] = 0x00;
	ECU.AutoGear.data[7] = 0x00;
	ECU.AutoGear.len = 8;
	
    if(speed <= 5)
    {
		canbus_ctrl_dec(1);
		deceleration = 0.8;
		while(speed >= 0.125)
		{
			usleep(200000);
		}

        while(gear != nowGear)
        {
			switch(gearDst)
			{
			case PARK:
				ECU.AutoGear.data[0] = PARK;    //P
				nowGear = PARK;
				break;
			case REVERSE:
				ECU.AutoGear.data[0] = REVERSE;    //R
				nowGear = REVERSE;
				break;
			case NEUTRAL:
				ECU.AutoGear.data[0] = NEUTRAL;    //N
				nowGear = NEUTRAL;
				break;
			case DRIVE:
				ECU.AutoGear.data[0] = DRIVE;    //D
				nowGear = DRIVE;
				break;
			default:
				break;
			}
			write(s, &(ECU.AutoGear), 16);
			printf("Gear ON! arduino gear:%d, car gear:%d\n", nowGear, gear);
			usleep(20000);
        }
		canbus_ctrl_dec(0);
    }
}
/* ========================================== */

/* = = = = = = = = = = = = = = = = = = = = = 油門踏板 = = = = = = = = = = = = = = = = = = = = = */
void canbus_ctrl_pedal(double pedalDst)
{
	if(pedalDst<0.75)
	{
		pedalDst = 0.75;
	}
	else if(pedalDst>20.3)
	{
		pedalDst = 20.3;
	}
    int pedal_v = int(pedalDst* (4095.0 / 4.96)); 
    int data0 = (pedal_v & 0xf00) >> 8;
    int data1 = (pedal_v & 0x0ff);
    ECU.AutoPedal.can_id = 0x251;
    ECU.AutoPedal.data[0] = data0;
    ECU.AutoPedal.data[1] = data1;
    ECU.AutoPedal.data[2] = 0x00;
    ECU.AutoPedal.data[3] = 0x00;
    ECU.AutoPedal.data[4] = 0x00;
    ECU.AutoPedal.data[5] = 0x00;
    ECU.AutoPedal.data[6] = 0x00;
    ECU.AutoPedal.data[7] = 0x00;
    ECU.AutoPedal.len = 8;
    ret = write(s, &(ECU.AutoPedal), 16);
	if (ret > 0) {
		TimeSyncMarkCanBrakeTxNs(TimeSyncNowNs());
	}
}
/* ========================================== */

/* = = = = = = = = = = = = = = = = = = = = = 方向盤 = = = = = = = = = = = = = = = = = = = = = */

inline void send_V_Rq_EPS_Ctrl()
{
	// unsigned char &V_Rq_EPS_Ctrl = canData.V_Rq_EPS_Ctrl;
	// printf("send_V_Rq_EPS_Ctrl: %x %x %x\n", ECU.EPAS1.data[0], ECU.EPAS1.data[1],ECU.EPAS1.data[3]);
	ECU.ECUMsg2.can_id = 0x201;
	ECU.ECUMsg2.data[0] = 0x02;	// V_Rq_EPS_Ctrl
	ECU.ECUMsg2.data[1] = ECU.EPAS1.data[0];
	ECU.ECUMsg2.data[2] = ECU.EPAS1.data[1];
	ECU.ECUMsg2.data[3] = 0x00;
	ECU.ECUMsg2.data[4] = 0x00;
	ECU.ECUMsg2.data[5] = 0x00;
	ECU.ECUMsg2.data[6] = 0x00;
	ECU.ECUMsg2.data[7] = 0x00;
	ECU.ECUMsg2.len = 8;

	// for(int i=0;i<20;i++)
	{
		ret = write(s, &(ECU.ECUMsg2), 16);
		// usleep(1000);
	}
}

inline void send_Rq_EPS_Ctrl()
{
#ifdef DEBUG
	printf("----------------send_Rq_EPS_Ctrl\n");
#endif
	ECU.ECUMsg2.can_id = 0x201;
	ECU.ECUMsg2.data[0] = 0x03;
	ECU.ECUMsg2.data[1] = ECU.EPAS1.data[0];
	ECU.ECUMsg2.data[2] = ECU.EPAS1.data[1];
	ECU.ECUMsg2.data[3] = 0x00;
	ECU.ECUMsg2.data[4] = 0x00;
	ECU.ECUMsg2.data[5] = 0x00;
	ECU.ECUMsg2.data[6] = 0x00;
	ECU.ECUMsg2.data[7] = 0x00;
	ECU.ECUMsg2.len = 8;

	ret = write(s, &(ECU.ECUMsg2), 16);
	if (ret > 0) {
		TimeSyncMarkCanSteerTxNs(TimeSyncNowNs());
	}
}

static inline void angleDecToCanbus(double angleIn, char canbusOut[2])
{
	short __angle__ = 10.0 * (angleIn + 900.0);
	canbusOut[0]= (__angle__& 0xff00)>>8;
	canbusOut[1]= (__angle__& 0x00ff);
}
inline void send_target_angle(double angle)
{
	// unsigned char &V_Rq_EPS_Ctrl = canData.V_Rq_EPS_Ctrl;
#ifdef DEBUG
	printf("----------------send_target_angle\n");
#endif
	char angleDst[2]={0};
	angleDecToCanbus(angle, angleDst);
	ECU.ECUMsg2.can_id = 0x201;
	ECU.ECUMsg2.data[0] = 0x03;	// V_Rq_EPS_Ctrl
	ECU.ECUMsg2.data[1] = angleDst[0];
	ECU.ECUMsg2.data[2] = angleDst[1];
	ECU.ECUMsg2.data[3] = 0x00;
	ECU.ECUMsg2.data[4] = 0x00;
	ECU.ECUMsg2.data[5] = 0x00;
	ECU.ECUMsg2.data[6] = 0x00;
	ECU.ECUMsg2.data[7] = 0x00;
	ECU.ECUMsg2.len = 8;

	ret = write(s, &(ECU.ECUMsg2), 16);
	if (ret > 0) {
		TimeSyncMarkCanSteerTxNs(TimeSyncNowNs());
	}
}

inline void send_max_speed()
{
	// unsigned char &V_Rq_EPS_Ctrl = canData.V_Rq_EPS_Ctrl;
#ifdef DEBUG
	printf("----------------send_max_speed\n");
#endif
	ECU.ECUMsg2.can_id = 0x202;
	ECU.ECUMsg2.data[0] = 0xFF;	// V_Rq_EPS_Ctrl
	ECU.ECUMsg2.data[1] = 0x00;
	ECU.ECUMsg2.data[2] = 0x00;
	ECU.ECUMsg2.data[3] = 0x00;
	ECU.ECUMsg2.data[4] = 0x00;
	ECU.ECUMsg2.data[5] = 0x00;
	ECU.ECUMsg2.data[6] = 0x00;
	ECU.ECUMsg2.data[7] = 0x00;
	ECU.ECUMsg2.len = 8;

	ret = write(s, &(ECU.ECUMsg2), 16);
	usleep(10000);
}



void *pth_canWheel(void *data)
{
	int controlled;
	double &steer = ptr_car->steer;
	double controllFactor;
	controlled = 0;
	t1.getStartTime();
	t1.getEndTime();

	while(SteerCtrlSwitch)
	{
		send_max_speed();
		int EPS_Sta_Available = ECU.EPAS1.data[5] & 0xc0;

		switch(EPS_Sta_Available)
		{
		case 0x00:
			t1.getEndTime();
			if(controlled == 1) goto clean_up;
			
			if((t1.diffTime())>0.02)
			{
				send_V_Rq_EPS_Ctrl();
				t1.getStartTime();
			}
			break;

		case 0x40:
			t1.getEndTime();
			if(controlled == 1) goto clean_up;

			if((t1.diffTime())>0.02)
			{
				send_Rq_EPS_Ctrl();
				controlled = 0;
				t1.getStartTime();
			}
			break;
		case 0x80:
			if(controlled == 0)
			{
				t1.getEndTime();
				if((t1.diffTime())>0.02)
				{
					send_Rq_EPS_Ctrl();
					t1.getStartTime();
				}
				controlled = 1;
			}
			else
			{
				t1.getEndTime();

				double finalAngle = 0.0;
				if((t1.diffTime()) <= 0.02) continue;

				/* 根據模式設定控制因子 */
				if(steerCtrlMode == WM_SET_ANGLE)
				{
					controllFactor = (targetAngle - steer);
				}
				else if(steerCtrlMode == WM_SET_ANGULAR_VELO)
				{
					controllFactor = targetAngle * t1.diffTime()*2;
				}

				if(controllFactor >= 95) controllFactor = 94;
				else if(controllFactor <= -95) controllFactor = -94;

				finalAngle = steer + controllFactor;

				if(finalAngle>=505) finalAngle = 504;
				else if(finalAngle<=-505) finalAngle = -504;


				send_Rq_EPS_Ctrl();
				send_target_angle(finalAngle);
				t1.getStartTime();
			}
			break;
		}
	}
clean_up:
	controlled = 0;
	SteerCtrlSwitch = 0;
	controllFactor = 0.0;
	return NULL;

}

void canbus_ctrl_steer(int SW)
{
	switch(SW)
	{
	case 0:
		SteerCtrlSwitch = 0;
		pthread_join(canWheel, NULL);
		std::cout<<"Successful. \"Control steering wheel\" module has been stopped.";
		break;

	case 1:
		if(SteerCtrlSwitch == 0)
		{
			SteerCtrlSwitch = 1;
			std::cout << "Boot \"Control steering wheel\" module." << std::endl;
			pthread_create(&canWheel, NULL, pth_canWheel, (void *)ptr_car);
		}
		else if(SteerCtrlSwitch == 1)
		{
			std::cout << "\"Control steering wheel\" module has been booted" << std::endl;
		}
		break;
	}
}


//-----------------------------------------------------------------------------------------------------

void canbus_recv(CAR &car)
{
	ptr_car = &car;

	double _tire_ = 0.0;
	double _steerWheel_ = 0.0;

	// FILE *fileIn = fopen("steer_wheel_to_tire.txt", "rt");
	// if(fileIn == NULL)
	// {
	// 	perror("steer_wheel_to_tire.txt");
	// 	exit(1);
	// }
	// fscanf(fileIn, "steerWheel:%lf\n", &_steerWheel_);
	// fscanf(fileIn, "tire:%lf", &_tire_);
	// fclose(fileIn);

	ptr_car->tire_max = _tire_;
	ptr_car->wheel_max = _steerWheel_;
	////////////////////////////////////////////////////
	s = ECU.createSocket();
	if (s < 0)
	{
		perror("socket");
	}
	ECU.setInterface("can0");
	ECU.linkToAddr();
	// ECU.setFilter();
	ECU.setSocket();
	if (ECU.bindSocket())
	{
		perror("bind");
		exit(1);
	}
	iov.iov_base = &canFrame;
	msg.msg_name = &(ECU.getSocketAddr());
	msg.msg_iov = &iov;
	msg.msg_iovlen = 1;
	msg.msg_control = &ctrlmsg;

		//=======can1===============
	s_can1 = ECU_can1.createSocket();
	if (s_can1 < 0)
	{
		perror("建立 CAN1 套接字");
	}

	ECU_can1.setInterface("can1");
	ECU_can1.linkToAddr();
	ECU_can1.setSocket();
	if (ECU_can1.bindSocket())
	{
		perror("綁定 CAN1 套接字");
		exit(1);
	}
	//=======can1===============


	////////////////////////////////////////////////////
	pthread_create(&canRecvSend, NULL, pth_canRecv, (void *)ptr_car);
	pthread_create(&can1_pth, NULL, pth_can1, (void *)ptr_car); //=======can1===============
	////////////////////////////////////////////////////
}

void wait_pthread()
{
	pthread_join(canRecvSend, NULL);
	pthread_join(can1_pth, NULL);   // <- 補上
	pthread_join(canWheel, NULL);
	pthread_join(canDec, NULL);
}

void canbus_set_steering_tx_enabled(int enabled)
{
	if (enabled == 0)
	{
		canbus_ctrl_steer(0);
		return;
	}

	send_V_Rq_EPS_Ctrl();
	usleep(10000);
	send_Rq_EPS_Ctrl();
	usleep(10000);
	send_Rq_EPS_Ctrl();
	canbus_ctrl_steer(1);
}
