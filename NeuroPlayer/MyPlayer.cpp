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

ofstream debug("nnetdebug.txt", ofstream::app);

enum ElementType { TFOOD, TENEMY, TBLOCK, TPOISON, TPLAYER, TCOUNT };


const int SENSOR_COUNT = 20;
const int INPUT_NEURON_COUNT = SENSOR_COUNT * 2;
//const int OUTPUT_NEURON_COUNT = 1;

enum Actions { FORWARD, BACKWARD, LEFTSTEP, RIGHTSTEP, COUNT = SENSOR_COUNT };

const int OUTPUT_NEURON_COUNT = SENSOR_COUNT;

const int HIDDEN_NEURON_COUNT = 100;
const int LAYER_COUNT = 3;
const int EPOCH_COUNT = 20;
const int TRAINSET_SIZE = 10;
const int MAX_EXPERIENCE = 20000;

int TRAINING_TICK = 5;

const int RANDOM_TRAINING_PERIOD = 1;
const int ONLY_RUNNING = 400000;
const int RANDOM_RATE = 100;
int ACTIVATION_FUNCTIONS[] = {TANH, TANH, LINE};
int LAYER_SIZES[] = { HIDDEN_NEURON_COUNT, HIDDEN_NEURON_COUNT, OUTPUT_NEURON_COUNT };

double step_angle;

const double DISTANCE_NORMA = 1.0;

const double QL_GAMMA = 0.95;
const double ALPHA = 0.5;

Matrix2D vr(OUTPUT_NEURON_COUNT, 1);
Matrix2D TrainSet;
TrainParams Params;
deque<vector<double>> VTrainSet[Actions::COUNT];
deque<vector<double>> QTrainSet;
vector<Net> nets;

Net net, target_prediction_net;

void QLearningMove(MyPlayer *me);
void CalcQEstimationForCurrentState();
int GetNextAction();
void UpdateTrainingSet(int NetId, Matrix2D &inp, Matrix2D &outp);
void RMSMove(MyPlayer *me);
void setInput(Matrix2D &inputs, Player *me, const int eyes, World *w);
pair<double, ElementType> getDistanceOnWall(Player *me, World *w, double angle);
bool check(Element *player, Element *elem, double angle);
bool check(Element *player, double x, double y, double r, double angle);
void DoAction(MyPlayer* me, Actions action);

double DistToWall(MyPlayer* me) { 
	double X = me->GetX(), Y = me->GetY();
	double W = me->GetWorld()->GetWidth(), H = me->GetWorld()->GetHeight();
	return min(min(X, W - X),min(Y, H - Y)) - me->GetR(); 
}

int tick = -1;
int lastAction = 0;
double lastReward = 0.0, lastLastReward = 0.0;
Matrix2D inputs(INPUT_NEURON_COUNT, 1), lastInputs(INPUT_NEURON_COUNT, 1);
Matrix2D output(OUTPUT_NEURON_COUNT, 1), maxlastOut(OUTPUT_NEURON_COUNT, 1);

bool check(Element *player, double x, double y, double r, double angle, double step_angle)
{
	double xfe = cos(angle);
	double yfe = sin(angle);
	double xse = cos(angle + step_angle);
	double yse = sin(angle + step_angle);
	double xo = x - player->GetX();
	double yo = y - player->GetY();

	//angle first eye - object
	double acos_val = (xfe*xo + yfe*yo) / sqrt(xo*xo + yo*yo);
	double angleA = acos(acos_val);
	if (angleA < 0.0)
		angleA += M_PI;
	//angle second eye - object
	acos_val = (xse*xo + yse*yo) / sqrt(xo*xo + yo*yo);
	double angleB = acos(acos_val);
	if (angleB < 0.0)
		angleB += M_PI;

	return angleA - step_angle <= DBL_EPSILON && angleB - step_angle <= DBL_EPSILON;
}




void MyPlayer::Init()
{
	SetName(L"NeuroPlayer");
	SetEyeCount(SENSOR_COUNT);
	step_angle = M_PI*2.0 / double(SENSOR_COUNT);
	//nets.resize(Actions::COUNT, Net(INPUT_NEURON_COUNT, LAYER_COUNT, ACTIVATION_FUNCTIONS, LAYER_SIZES));
	net = Net(INPUT_NEURON_COUNT, LAYER_COUNT, ACTIVATION_FUNCTIONS, LAYER_SIZES);
	target_prediction_net = net;
	Params.Error = 0.001;
	Params.MinGrad = 0.00001;
	Params.NumEpochs = EPOCH_COUNT;
	Params.Rate = 0.0005;
}

void MyPlayer::Move()
{
	QLearningMove(this);
}



void UpdateTrainingSet(int NetId, Matrix2D &inp, Matrix2D &outp) 
{
	VTrainSet[NetId].emplace_back(vector<double>(INPUT_NEURON_COUNT + OUTPUT_NEURON_COUNT, 0.0));
	if (VTrainSet[NetId].size() > MAX_EXPERIENCE)
		VTrainSet[NetId].pop_front();
	for (int i = 1; i <= INPUT_NEURON_COUNT; ++i) {
		VTrainSet[NetId].back()[i - 1] = inp(i, 1);
	}
	for (int i = 1; i <= OUTPUT_NEURON_COUNT; ++i)
		VTrainSet[NetId].back()[i + INPUT_NEURON_COUNT - 1] = outp(i, 1);

	Matrix2D CurTrainset(min(VTrainSet[NetId].size(), TRAINSET_SIZE), INPUT_NEURON_COUNT + OUTPUT_NEURON_COUNT);
	for (int i = 0; i < min(VTrainSet[NetId].size(), TRAINSET_SIZE); ++i) {
		int cur_idx = dst(eng) % VTrainSet[NetId].size();
		for (int j = 1; j <= INPUT_NEURON_COUNT; ++j) {
			CurTrainset(i + 1, j) = VTrainSet[NetId][cur_idx][j - 1];
		}
		for (int j = 1; j <= OUTPUT_NEURON_COUNT; ++j)
			CurTrainset(i + 1, j + INPUT_NEURON_COUNT) = VTrainSet[NetId][cur_idx][j + INPUT_NEURON_COUNT - 1];
	}
	nets[NetId].TrainSet(CurTrainset);
}

void UpdateTrainingSet(Matrix2D &inp, Matrix2D &outp)
{
	QTrainSet.emplace_back(vector<double>(INPUT_NEURON_COUNT + OUTPUT_NEURON_COUNT, 0.0));
	if (QTrainSet.size() > MAX_EXPERIENCE)
		QTrainSet.pop_front();
	for (int i = 1; i <= INPUT_NEURON_COUNT; ++i) {
		QTrainSet.back()[i - 1] = inp(i, 1);
	}
	for (int i = 1; i <= OUTPUT_NEURON_COUNT; ++i)
		QTrainSet.back()[i + INPUT_NEURON_COUNT - 1] = outp(i, 1);

	if (QTrainSet.size() < TRAINSET_SIZE) return;

	Matrix2D CurTrainset(min(QTrainSet.size(), TRAINSET_SIZE), INPUT_NEURON_COUNT + OUTPUT_NEURON_COUNT);
	vector<short> used(QTrainSet.size(), 0);
	int cur_idx;
	for (int i = 0; i < min(QTrainSet.size(), TRAINSET_SIZE); ++i) {
		cur_idx = dst(eng) % QTrainSet.size();
		while (used[cur_idx]) cur_idx = dst(eng) % QTrainSet.size();
		used[cur_idx] = true;
		for (int j = 1; j <= INPUT_NEURON_COUNT; ++j) {
			CurTrainset(i + 1, j) = QTrainSet[cur_idx][j - 1];
		}
		for (int j = 1; j <= OUTPUT_NEURON_COUNT; ++j)
			CurTrainset(i + 1, j + INPUT_NEURON_COUNT) = QTrainSet[cur_idx][j + INPUT_NEURON_COUNT - 1];
	}
	net.TrainSet(CurTrainset);
}


double curFullness = 0.0, lastFullness = 0.0, curHealth = 0.0, lastHealth = 0.0;

void QLearningMove(MyPlayer *me)
{
	tick++;
	int rnd = dst(eng);
	int action = -1;
	
	// Ќаграда за предыдущий шаг
	double wallReward = -2*exp(-DistToWall(me) / 80.0);
	curFullness = me->GetFullness();
	curHealth = me->GetHealth();
	lastReward = 10.0*(curFullness - lastFullness) + 5.0*(curHealth - lastHealth) + wallReward;
	lastFullness = curFullness;
	lastHealth = curHealth;

	/////////////////////////////////////////////////////////////////////
	// Ќовое состо€ние сенсоров
	setInput(inputs, me, SENSOR_COUNT, me->GetWorld());
	/////////////////////////////////////////////////////////////////////

	// ¬ычисление оценки выгоды хода и выбор действи€
	CalcQEstimationForCurrentState();
	action = GetNextAction();

	//  ќбновление обучающей выборки
	if (tick)
		//UpdateTrainingSet(lastAction, lastInputs, maxlastOut);
		UpdateTrainingSet(lastInputs, maxlastOut);

	// ’одим случайно в первый тик, обучающий период и каждый TRAINING_TICK тик
	if ((!tick || tick <= RANDOM_TRAINING_PERIOD || tick % RANDOM_RATE == 0) && tick < ONLY_RUNNING)
	{
		action = rnd % Actions::COUNT;
	}	

	// ќбучаем сеть каждый TRAINING_TICK тик
	if (tick && tick % TRAINING_TICK == 0 && tick < ONLY_RUNNING)
	{
		vector<double> training_error;
		//Params.NumEpochs = max(1, EPOCH_COUNT * (1.0 - tick * 1.0 / (ONLY_RUNNING)));
		net.RMSPropTrain(Params, training_error);
		auto WB = net.WeightsBiases();
		auto TargetWB = target_prediction_net.WeightsBiases();
		auto UpdatedWB = (1 - ALPHA)*TargetWB + ALPHA*WB;
		target_prediction_net.WeightsBiases(UpdatedWB);
		target_prediction_net.WBNetToLayers();
	}

	// ¬ыполн€ем действие и сохран€ем текущие действие и состо€ние дл€ следующего шага
	DoAction(me, (Actions)action);

	lastAction = action;
	lastInputs = inputs;
	lastLastReward = lastReward;

}

void CalcQEstimationForCurrentState() 
{
	// ¬ычисление оценки полезности дл€ нового состо€ни€ сенсоров
	double Q = -DBL_MAX;
	target_prediction_net.Simulate(inputs, output);
	for (int i = 0; i < Actions::COUNT; ++i)
	{
		Q = max(output(i + 1, 1), Q);
	}
	output(lastAction + 1, 1) = Q*QL_GAMMA + lastReward;
	maxlastOut = output;
}

int GetNextAction()
{
	int action = -1;
	// ¬ычисление оценки полезности дл€ нового состо€ни€ сенсоров
	double Q = -DBL_MAX;
	net.Simulate(inputs, output);
	for (int i = 0; i < Actions::COUNT; ++i)
	{
		double curQ = output(i + 1, 1);
		if (Q < curQ)
		{
			Q = curQ;
			action = i;
		}
	}
	return action;
}


void RMSMove(MyPlayer *me) 
{
	int rnd = dst(eng);
	int action = rnd % Actions::COUNT;

	if (tick >= ONLY_RUNNING) {
		setInput(inputs, me, SENSOR_COUNT, me->GetWorld());
		lastReward = -DBL_MAX;
		for (int i = 0; i < nets.size(); ++i)
		{
			nets[i].Simulate(inputs, output);
			double curReward = output(1, 1);
			if (lastReward < curReward)
			{
				lastReward = curReward;
				action = i;
			}
		}
		DoAction(me, (Actions)action);
		return;
	}

	tick++;

	/////////////////////////////////////////////////////////////////////
	setInput(inputs, me, SENSOR_COUNT, me->GetWorld());
	/////////////////////////////////////////////////////////////////////
	lastReward = me->GetFullness() + me->GetHealth();

	if (tick)
	{
		VTrainSet[lastAction].emplace_back(vector<double>(INPUT_NEURON_COUNT + OUTPUT_NEURON_COUNT, 0.0));
		if (VTrainSet[lastAction].size() > TRAINSET_SIZE)
			VTrainSet[lastAction].pop_front();
		for (int i = 1; i <= INPUT_NEURON_COUNT; ++i)
			VTrainSet[lastAction].back()[i - 1] = lastInputs(i, 1);
		for (int i = 1; i <= OUTPUT_NEURON_COUNT; ++i)
			VTrainSet[lastAction].back()[i + INPUT_NEURON_COUNT - 1] = lastReward;
		nets[lastAction].TrainSet(Matrix2D(VTrainSet[lastAction]));
		if (tick <= RANDOM_TRAINING_PERIOD || tick % TRAINING_TICK == 0)
		{
			vector<double> training_error;
			nets[lastAction].RMSPropTrain(Params, training_error);
		}
	}

	if (tick <= RANDOM_TRAINING_PERIOD || tick % TRAINING_TICK == 0)
	{
		action = rnd % Actions::COUNT;
	}
	else
	{
		double Q = -DBL_MAX;
		for (int i = 0; i < nets.size(); ++i)
		{
			nets[i].Simulate(inputs, output);
			double curQ = output(1, 1);

			if (Q < curQ)
			{
				Q = curQ;
				action = i;
			}
		}
	}

	lastInputs = inputs;
	DoAction(me, (Actions)action);
	lastAction = action;
}

void setInput(Matrix2D &inputs, Player *me, const int eyes, World *w)
{
	double angle = M_PI / 100.0;

	vector<pair<double, ElementType>> sensors(eyes, make_pair(DBL_MAX, ElementType::TCOUNT));
	int eye = 0;
	while (angle < 2 * M_PI + M_PI / 100.0)
	{
		double min_dist = DBL_MAX;
		double dist = DBL_MAX;
		ElementType cur_type = ElementType::TENEMY;
		int life_time = INT_MAX;

		//food
		for each (auto cur1 in w->GetFood())
		{
			dist = me->GetDistanceTo(cur1);
			//debug << cur->GetX() << " " << cur->GetY() << endl;
			if (check(me, cur1, angle) && dist < min_dist)
			{
				min_dist = dist;
				life_time = cur1->GetLifeTime();
				if (cur1->IsDamageBonus())
					cur_type = ElementType::TPOISON;
				else cur_type = ElementType::TFOOD;
			}

		}

		//players
		for each (auto cur2 in w->GetPlayers())
		{
			if (cur2 == me) continue;
			if (check(me, cur2, angle) && dist < min_dist)
			{
				min_dist = dist;
				cur_type = ElementType::TENEMY;
				life_time = cur2->GetSpeed();
			}

		}

		//block
		for each (auto cur3 in w->GetBlocks())
		{
			dist = me->GetDistanceTo(cur3);
			if (check(me, cur3, angle) && dist < min_dist)
			{
				min_dist = dist;
				cur_type = ElementType::TBLOCK;
			}
		}


		auto res = getDistanceOnWall(me, w, angle);

		if (min_dist > res.first)
		{
			min_dist = res.first;
			cur_type = res.second;
		}
		
		inputs(eye + 1, 1) = min_dist / DISTANCE_NORMA;
		inputs(eye + 2, 1) = cur_type;
		//inputs(eye + 3, 1) = life_time;

		//debug << "eye " << min_dist << " " << elty[cur_type] << endl;
		eye += 2;
		angle += step_angle;
	}
}

pair<double, ElementType> getDistanceOnWall(Player *me, World *w, double angle)
{
	double x0 = me->GetX();
	double y0 = me->GetY();
	double k = -tan(angle);

	double min_dist = DBL_MAX;
	
	//пересечение только с горизонтальными сторонами
	if (abs(angle - M_PI / 2) <= DBL_EPSILON) min_dist = min(min_dist, me->GetY());
	else if (abs(angle - 3 * M_PI / 2) <= DBL_EPSILON) min_dist = min(min_dist, w->GetHeight() - me->GetY());
	//пересечение только с вертилкальными сторонами
	else if (abs(angle) <= DBL_EPSILON) min_dist = min(min_dist, w->GetWidth() - me->GetX());
	else if (abs(angle - M_PI) <= DBL_EPSILON) min_dist = min(min_dist, me->GetX());
	//случайные пересечени€
	else
	{
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
		if (x >= -EPS && x <= w->GetWidth() + EPS && abs(cosv - 1.0) <= EPS)
			min_dist = min(min_dist, sqrt((x - x0)*(x - x0) + (y - y0)*(y - y0)));
		
		//y=geth
		y = w->GetHeight();
		x = (y - b) / k;
		vx = x - x0, vy = y - y0;
		cosv = (origin_vx*vx + origin_vy * vy) / sqrt(vx*vx + vy*vy);
		if (x >= -EPS && x <= w->GetWidth() + EPS && abs(cosv - 1.0) <= EPS)
			min_dist = min(min_dist, sqrt((x - x0)*(x - x0) + (y - y0)*(y - y0)));

		//x=0
		x = 0.0;
		y = k*x + b;
		vx = x - x0, vy = y - y0;
		cosv = (origin_vx*vx + origin_vy * vy) / sqrt(vx*vx + vy*vy);
		if (y >= -EPS && y <= w->GetHeight() + EPS && abs(cosv - 1.0) <= EPS)
			min_dist = min(min_dist, sqrt((x - x0)*(x - x0) + (y - y0)*(y - y0)));

		//x=getw
		x = w->GetWidth();
		y = k*x + b;
		vx = x - x0, vy = y - y0;
		cosv = (origin_vx*vx + origin_vy * vy) / sqrt(vx*vx + vy*vy);
		if (y >= -EPS && y <= w->GetHeight() + EPS && abs(cosv - 1.0) <= EPS)
			min_dist = min(min_dist, sqrt((x - x0)*(x - x0) + (y - y0)*(y - y0)));
	}

	return make_pair(min_dist / DISTANCE_NORMA, ElementType::TBLOCK);
}

bool check(Element *player, Element *elem, double angle)
{
	return check(player, elem->GetX(), elem->GetY(), elem->GetR(), angle);
	//return check(player, elem->GetX(), elem->GetY(), elem->GetR(), angle, step_angle);
}

bool check(Element *player, double x, double y, double r, double angle)
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

void DoAction(MyPlayer* me, Actions action)
{
	double angle = 2 * M_PI / SENSOR_COUNT;
	double start_angle = M_PI / 100;
	//me->MoveTo(me->GetX() + 600.0 * cos(start_angle + angle*action + 0.5*angle), me->GetY() - 600.0 * sin(start_angle + angle*action + 0.5*angle));
	me->MoveTo(me->GetX() + 600.0 * cos(start_angle + angle*action), me->GetY() - 600.0 * sin(start_angle + angle*action));
	
	/*switch (action)
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
	}*/
}