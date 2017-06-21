#include <vector>
#include <iostream>
#include <algorithm>
#include "World.h"
#include "MyPlayer.h"
#include "NNet.h"
#include <fstream>
#include <ctime>
#include <exception>
#include <climits>
#include <omp.h>
#include <random>
#include <memory>
#include <random>
#include <queue>

using namespace std;

const double EPS = 1e-5;

random_device rd2;
mt19937 eng2(rd2());
uniform_int_distribution<> dst2(1, 50000);

enum Actions { FORWARD, BACKWARD, LEFTSTEP, RIGHTSTEP, COUNT };

const int SENSOR_COUNT = 8;
const int INPUT_NEURON_COUNT = SENSOR_COUNT * 2;
const int OUTPUT_NEURON_COUNT = 1;
const int HIDDEN_NEURON_COUNT = INPUT_NEURON_COUNT*1.5;
const int LAYER_COUNT = 2;
const int EPOCH_COUNT = 100;
const int TRAINSET_SIZE = 50;
const int TRAINING_TICKS = 1;
const int RANDOM_TRAINING = 1;
int ACTIVATION_FUNCTIONS[] = {TANH, LINE};
int LAYER_SIZES[] = { HIDDEN_NEURON_COUNT, OUTPUT_NEURON_COUNT };

Matrix2D vr(OUTPUT_NEURON_COUNT, 1);

Matrix2D TrainSet;

TrainParams Params;

deque<vector<double>> VTrainSet[4];


void DoAction(MyPlayer* me, Actions action)
{
	switch (action)
	{
	case Actions::FORWARD:
		me->StepForward();
		break;

	case Actions::BACKWARD:
		me->StepBackward();
		break;

	case Actions::LEFTSTEP:
		me->StepLeft();
		break;

	case Actions::RIGHTSTEP:
		me->StepRight();
		break;

	default:
		break;
	}
}

SYSTEMTIME st;
bool FirstStep = true;
vector<Net> nets;
//NeuroNet::ElmanNetwork net;
void MyPlayer::Init()
{
	SetName(L"NeuroPlayer");
	SetEyeCount(SENSOR_COUNT);
	nets.resize(Actions::COUNT, Net(INPUT_NEURON_COUNT, LAYER_COUNT, ACTIVATION_FUNCTIONS, LAYER_SIZES));
	Params.Error = 0.001;
	Params.MinGrad = 0.00001;
	Params.NumEpochs = EPOCH_COUNT;
	Params.Rate = 0.001;
}

bool check(Element *player, double x, double y, double angle, double step_angle)
{
	double vx = 10 * cos(angle);
	double vy = 10 * sin(angle);
	double vx2 = x - player->GetX();
	double vy2 = y - player->GetY();
	double cosa = (vx*vx2 + vy*vy2) *1.0 /
		(
			sqrt(vx*vx + vy*vy)
			*sqrt(vx2*vx2 + vy2*vy2)
			);
	cosa = max(-1.0 + EPS, cosa);
	cosa = min(1.0 - EPS, cosa);
	double angleB = acos(cosa);
	if (angleB < -FLT_MAX)
		int y = 0;
	//debug << " in check cosa = " << cosa << " angle = " << angleB << " result = " << (abs(angleB) <= step_angle / 2.0);
	return abs(angleB) <= step_angle / 2.0 + EPS;
}

bool check(Element *player, double x, double y, double r, double angle, double step_angle)
{
	double x0 = player->GetX(), y0 = player->GetY();
	double x2 = x0 + 1.0 * cos(angle), y2 = y0 - 1.0*sin(angle);
	double k = (y2 - y0) / (x2 - x0), b = -x0*k + y0;
	if (abs(x2 - x0) <= EPS) {
		double Ds = r*r - (y0 - y)*(y0 - y);
		if (Ds >= 0.0) {
			double cross_x1 = sqrt(Ds) + x, cross_y1 = y0;
			double cosv = ((cross_x1 - x0)*(x2 - x0) + (cross_y1 - y0)*(y2 - y0)) / (sqrt((cross_x1 - x0)*(cross_x1 - x0) + (cross_y1 - y0)*(cross_y1 - y0)));
			if (cosv >= -EPS) {
				return true;
			}
			return false;
		}
	}
	else {
		double Ds = (x - k*(b - y))*(x - k*(b - y)) - (1 + k*k)*(x*x + (b - y)*(b - y) - r*r);
		if (Ds >= 0.0) {
			double cross_x1 = ((x - k*(b - y)) + sqrt(Ds)) / (1 + k*k), cross_y1 = k*cross_x1 + b;
			double cosv = ((cross_x1 - x0)*(x2 - x0) + (cross_y1 - y0)*(y2 - y0)) / (sqrt((cross_x1 - x0)*(cross_x1 - x0) + (cross_y1 - y0)*(cross_y1 - y0)));
			if (cosv >= -EPS) {
				return true;
			}
			return false;
		}
	}
	return false;
}



bool check(Element *player, Element *elem, double angle, double step_angle)
{
	//return check(player, elem->GetX(), elem->GetY(), angle, step_angle);
	return check(player, elem->GetX(), elem->GetY(), elem->GetR(), angle, step_angle);
}


enum ElementType { TFOOD, TENEMY, TBLOCK, TCOUNT };
string elty[] = { "FOOD", "ENEMY", "BLOCK", "COUNT" };

pair<double, ElementType> getDistanceOnWall(Player *me, World *w, double angle, double step_angle)
{
	//debug << "start" << endl;
	double x0 = me->GetX();
	double y0 = me->GetY();
	double k = -tan(angle);
	double angleB;
	//debug << "inited" << endl;

	//Y = 0, x = 0..getw
	double dist = DBL_MAX;
	double min_dist = DBL_MAX;
	ElementType cur_type = ElementType::TENEMY;
	//debug << "before if" << endl;
	if (abs(angle - M_PI / 2) <= DBL_EPSILON)
	{
		//debug << "in first if" << endl;
		//пересечение только с горизонтальными сторонами
		dist = min(dist, me->GetY());
		if (min_dist > dist)
		{
			min_dist = dist;
			cur_type = ElementType::TBLOCK;
		}
	}
	else if (abs(angle - 3 * M_PI / 2) <= DBL_EPSILON)
	{
		//debug << "in first if" << endl;
		//пересечение только с горизонтальными сторонами
		dist = min(dist, w->GetHeight() - me->GetY());
		if (min_dist > dist)
		{
			min_dist = dist;
			cur_type = ElementType::TBLOCK;
		}
	}
	else if (abs(angle) <= DBL_EPSILON)
	{
		//debug << "in second if" << endl;
		//пересечение только с вертилкальными сторонами
		dist = min(dist, w->GetWidth() - me->GetX());
		if (min_dist > dist)
		{
			min_dist = dist;
			cur_type = ElementType::TBLOCK;
		}
	}
	else if (abs(angle - M_PI) <= DBL_EPSILON)
	{
		//debug << "in second if" << endl;
		//пересечение только с вертилкальными сторонами
		dist = min(dist, me->GetX());
		if (min_dist > dist)
		{
			min_dist = dist;
			cur_type = ElementType::TBLOCK;
		}
	}
	else
	{
		//debug << "in else" << endl;
		//случайные пересечения
		/*
		система:
		y = 0, при 0.0 <= x <= getw();
		y = geth(), при 0.0 <= x <= getw();

		x = 0, при 0.0 <= y <= geth();
		x = getw(), при 0.0 <= y <= geth();
		*/

		//y = 0
		double b = y0 - k*x0;
		double y = 0.0;
		double x = (y - b) / k;

		double origin_vx = cos(angle), origin_vy = -sin(angle);

		double vx = x - x0, vy = y - y0;
		double cosv = (origin_vx*vx + origin_vy * vy) / sqrt(vx*vx + vy*vy);
		//debug << "1xy b = " << b << " x = " << x << " y = " << y;
		if (x >= -EPS && x <= w->GetWidth() + EPS && /*check(me, x, y, angle, step_angle)*/ abs(cosv - 1.0) <= EPS)
		{
			//debug << " dist = " << sqrt((x - x0)*(x - x0) + (y - y0)*(y - y0));
			dist = min(dist, sqrt((x - x0)*(x - x0) + (y - y0)*(y - y0)));
		}
		//debug << endl;
		//y=geth
		y = w->GetHeight();
		x = (y - b) / k;
		vx = x - x0, vy = y - y0;
		cosv = (origin_vx*vx + origin_vy * vy) / sqrt(vx*vx + vy*vy);
		//debug << "2xy b = " << b << " x = " << x << " y = " << y;
		if (x >= -EPS && x <= w->GetWidth() + EPS && abs(cosv - 1.0) <= EPS)
		{
			//debug << " dist = " << sqrt((x - x0)*(x - x0) + (y - y0)*(y - y0));
			dist = min(dist, sqrt((x - x0)*(x - x0) + (y - y0)*(y - y0)));
		}
		//debug << endl;

		//x=0
		x = 0.0;
		y = k*x + b;
		vx = x - x0, vy = y - y0;
		cosv = (origin_vx*vx + origin_vy * vy) / sqrt(vx*vx + vy*vy);
		//debug << "3xy b = " << b << " x = " << x << " y = " << y;
		if (y >= -EPS && y <= w->GetHeight() + EPS && abs(cosv - 1.0) <= EPS)
		{
			//debug << " dist = " << sqrt((x - x0)*(x - x0) + (y - y0)*(y - y0));
			dist = min(dist, sqrt((x - x0)*(x - x0) + (y - y0)*(y - y0)));
		}
		//debug << endl;

		//x=getw
		x = w->GetWidth();
		y = k*x + b;
		vx = x - x0, vy = y - y0;
		cosv = (origin_vx*vx + origin_vy * vy) / sqrt(vx*vx + vy*vy);
		//debug << "4xy b = " << b << " x = " << x << " y = " << y;
		if (y >= -EPS && y <= w->GetHeight() + EPS && abs(cosv - 1.0) <= EPS)
		{
			//debug << " dist = " << sqrt((x - x0)*(x - x0) + (y - y0)*(y - y0));
			dist = min(dist, sqrt((x - x0)*(x - x0) + (y - y0)*(y - y0)));
		}
		//debug << endl;

		if (min_dist > dist)
		{
			min_dist = dist;
			cur_type = ElementType::TBLOCK;
		}
	}
	//debug << "return" << endl;

	//debug << min_dist << " " << elty[cur_type] << endl;
	return make_pair(min_dist, cur_type);
}

void setInput(Matrix2D &inputs, Player *me, const int eyes, World *w)
{
	const double step_angle = M_PI*2.0 / eyes;
	double angle = M_PI / 100.0;

	vector<pair<double, ElementType>> sensors(eyes, make_pair(DBL_MAX, ElementType::TCOUNT));
	int eye = 0;
	while (angle < 2 * M_PI + M_PI / 100.0)
	{
		double min_dist = DBL_MAX;
		double dist = DBL_MAX;
		ElementType cur_type = ElementType::TENEMY;
		//food
		for each (auto cur in w->GetFood())
		{
			dist = me->GetDistanceTo(cur);
			//debug << cur->GetX() << " " << cur->GetY() << endl;
			if (check(me, cur, angle, step_angle) && dist < min_dist)
			{
				//debug << "food dist = " << dist << endl;
				min_dist = dist;
				cur_type = ElementType::TFOOD;
			}

		}

		//block
		for each (auto cur in w->GetBlocks())
		{
			dist = me->GetDistanceTo(cur);
			if (check(me, cur, angle, step_angle) && dist < min_dist)
			{
				min_dist = dist;
				cur_type = ElementType::TBLOCK;
			}
		}
		if (cur_type == TENEMY)
			int y = 0;


		auto res = getDistanceOnWall(me, w, angle, step_angle);
		if (res.second == TENEMY)
			int y = 0;

		if (min_dist > res.first)
		{
			min_dist = res.first;
			cur_type = res.second;
		}
		if (cur_type == TENEMY)
			int y = 0;
		inputs(eye + 1, 1) = min_dist;
		inputs(eye + 2, 1) = cur_type;

		//debug << "eye " << min_dist << " " << elty[cur_type] << endl;
		eye += 2;
		angle += step_angle;
	}


}

int lasttest = -1;
int tick = -1;
int lastAction = 0;
double Q, lastQ = 0.0;
Matrix2D inputs(INPUT_NEURON_COUNT, 1), LastInputs(INPUT_NEURON_COUNT, 1);
Matrix2D output(OUTPUT_NEURON_COUNT, 1);

void MyPlayer::Move()
{
	tick++;
	
	//Input order: My coordinates, Health, Fullness, Angle, Food coordinates, Enemy coordinates, Trap coordinates, Poison coordinates, Cornucopia coordinates, Block coordinates

	/////////////////////////////////////////////////////////////////////
	///////////////////////// InitFill input vector /////////////////////////
	setInput(inputs, this, SENSOR_COUNT, GetWorld());
	/////////////////////////////////////////////////////////////////////
	//debug << inputs << endl;
	vr(1, 1) = GetFullness();

	int rnd = dst2(eng2);
	int action = rnd % Actions::COUNT;
	if (!FirstStep)
	{
		/*int maxtests = 500;
		int mintest = 100;
		if (TrainingSet.size() > maxtests)
		{
			vector<NeuroNet::Problem> vp;
			int ct = mintest;
			while (ct--)
				vp.push_back(TrainingSet[dist(eng) % TrainingSet.size()]);
		}*/
		VTrainSet[lastAction].emplace_back(move(vector<double>(INPUT_NEURON_COUNT + OUTPUT_NEURON_COUNT)));
		if (VTrainSet[lastAction].size() > TRAINSET_SIZE)
			VTrainSet[lastAction].pop_front();
		for (int i = 1; i <= INPUT_NEURON_COUNT; ++i)
			VTrainSet[lastAction].back()[i - 1] = LastInputs(i, 1);
		for (int i = 1; i <= OUTPUT_NEURON_COUNT; ++i)
			VTrainSet[lastAction].back()[i + INPUT_NEURON_COUNT - 1] = vr(i, 1);
		nets[lastAction].TrainSet(Matrix2D(VTrainSet[lastAction]));
		if (tick <= RANDOM_TRAINING*TRAINING_TICKS ||  tick % 5 == 0)
		{
			//debug << "TICK " << tick << endl;
			vector<double> training_error;
			nets[lastAction].RMSPropTrain(Params, training_error);
		}
	}

	//	debug << "R = " << r << endl;
	/*if (tick <= TRAINING_TICKS) {
		if (tick <= TRAINING_TICKS / 4) {
			action = Actions::FORWARD;
		}
		else if (tick <= TRAINING_TICKS / 2) action = Actions::LEFTSTEP;
		else if (tick <= 3 * TRAINING_TICKS / 4) action = Actions::BACKWARD;
		else action = Actions::RIGHTSTEP;
	}
	else */
		if (tick <= RANDOM_TRAINING*TRAINING_TICKS || tick % 5 == 0)
	{
		//	debug << "RAND: " << endl;;

		Q = vr(1, 1);
		action = rnd % Actions::COUNT;
	}
	else
	{

		//debug << "Before Learn" << endl;
		Q = -DBL_MAX;
		for (int i = 0; i < nets.size(); ++i)
		{
			
			nets[i].Simulate(inputs, output);
			double curQ = output(1, 1);
			//debug << curQ << " " << i << endl;

			if (Q < curQ)
			{
				Q = curQ;
				action = i;
			}
		}
		//debug << "After Learn" << endl;
	}

	//debug << "SELECT " << Q << " " << action << endl << "=================================================================" << endl;

	LastInputs = inputs;
	DoAction(this, (Actions)action);
	FirstStep = false;

	lastAction = action;
}