#ifndef _MY_PLAYER
#define _MY_PLAYER
#pragma once

#include "Player.h"

#pragma comment (lib, "World.lib")

class MyPlayer : public Player {
public:
	void Init() override;
	void Move() override;
};

extern "C" __declspec (dllexport) MyPlayer* GetMyPlayer();
//MyPlayer* GetMyPlayer();
#endif