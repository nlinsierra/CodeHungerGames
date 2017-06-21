#ifndef _PLAYER
#define _PLAYER

#pragma once

#include "Element.h"
#include "World.h"
#include "Global.h"
#include <vector>

class World;
class  __declspec (dllexport) Player : public Element {
	friend class World;
private:
	std::wstring Name;
	double Angle;
	int Speed;
	int RotateSpeed;
	int Health;
	int Fullness; //�������
	std::vector<Player*> Enemies;
	bool HasGolograf;
	World* CurrentWorld;
	D3DCOLOR Color;
	int Idx;
	HMODULE PlayerLib;
	bool RunTime;
	bool TimeLimit;
	bool MemoryLimit;

	int EyeCount;
	std::vector<D3DCOLOR> EyeColors;

private:
	void Render(LPDIRECT3DDEVICE9 d3ddev);
	void RenderFull(LPDIRECT3DDEVICE9 d3ddev);

	void SetSpeed(int NewSpeed) { Speed = NewSpeed; };
	void SetRotateSpeed(int NewRotateSpeed) { RotateSpeed = NewRotateSpeed; };
	void SetHealth(int NewHealth) { Health = NewHealth; };
	void SetFullness(int NewFullness) { Fullness = NewFullness; };
	void SetAngle(double NewAngle) { Angle = NewAngle; };
	void AddEnemy(Player *EP) { Enemies.emplace_back(EP); };
	void ClearEnemies() { Enemies.clear(); };
	void SetHasGolograf(bool NewHasGolograf) { HasGolograf = NewHasGolograf; };
	void SetWorld(World *NewWorld) { CurrentWorld = NewWorld; };
	D3DCOLOR GetColor() { return Color; };
	void SetColor(D3DCOLOR NewColor) { Color = NewColor; };
public:
	Player() {
		Speed = 0; Health = 0; Fullness = 0; 
		HasGolograf = false; PlayerLib = nullptr; 
		RunTime = TimeLimit = MemoryLimit = false;
	};
	~Player() {};
	int GetSpeed() { return Speed; };
	int GetHealth() { return Health; };
	int GetFullness() { return Fullness; };
	bool GetHasGolograf() { return HasGolograf; };
	World* GetWorld() { return CurrentWorld; };
	std::wstring GetName() { return Name; };
	double GetAngle() { return Angle; };
	void SetName(std::wstring NewName) { Name = NewName; };

	void Rotate(double RotationAngle);
	
	double GetAngleTo(int ToX, int ToY);
	double GetAngleTo(Element *SomeObject);

	double GetDistanceTo(int ToX, int ToY);
	double GetDistanceTo(Element *SomeObject);

	void MoveTo(int ToX, int ToY);
	void MoveTo(Element *SomeObject);
	void Strike(Player *EP);
	void StepForward();
	void StepBackward();
	void StepLeft();
	void StepRight();

	void SetEyeCount(int c) { EyeCount = c; EyeColors.resize(EyeCount, D3DCOLOR_XRGB(255, 255, 255)); };
	int GetEyeCount() { return EyeCount; };

	void SetEyeColor(int i, D3DCOLOR col) { EyeColors[i] = col; };
	D3DCOLOR GetEyeColor(int i) { return EyeColors[i]; };

	virtual void Init() {};
	virtual void Move() {};
};

#endif

