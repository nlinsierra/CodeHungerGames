#include <windows.h>
#include <fstream>
#include <sstream>
#include <Processthreadsapi.h>
#include "global.h"
#include <ctime>
#include <iomanip>

using namespace std;
wofstream debugout("debug.txt",wofstream::app);
DWORD WINAPI DoAll(LPVOID lpParam);

HINSTANCE  inj_hModule;          //Injected Modules Handle
HWND       hWnd;            //Window Handle


bool CompileSolution(char *CompileCommand);
bool PerformCompilation();
void Compile();
char CurDir[MAX_STR_LEN];
#define DIRECTXPATH "C:\\Program Files (x86)\\Microsoft DirectX SDK (June 2010)\\"
#define MVSDIR "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\"
#define OUTDIR "players"
#define INCLUDEDIR "include"
#define SULUTIONDIR "solutions"

DWORD WINAPI DoAll(LPVOID lpParam)
{
	try {
		debugout << "Start Compile" << endl;
		Compile();
		debugout << "End Compile" << endl;
	}
	catch (...) {
		debugout << "Exception" << endl;
	}
	return 1;
	return 1;
}

BOOL APIENTRY DllMain(HMODULE hModule, DWORD  ul_reason_for_call, LPVOID lpReserved)
{
	//srand(time(nullptr));
	if (ul_reason_for_call == DLL_PROCESS_ATTACH) {
		inj_hModule = hModule;
		//CreateThread(0, NULL, ThreadProc, (LPVOID)L"Code hunger games challenge", NULL, NULL);
		DoAll(0);
	}
	return TRUE;
}

void Compile()
{
	GetCurrentDirectory(MAX_STR_LEN, CurDir);
	PerformCompilation();
}

// ������� ������� �������� ����������
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool CompileSolution(char *CompileCommand) {
	PROCESS_INFORMATION ProcessInf = { 0 };
	STARTUPINFO StartInf;
	stringstream LogMsg;
	char CompilerMsg[LOG_MSG_LEN] = "";
	HANDLE hPipeRead = NULL, hPipeWrite = NULL;
	DWORD Read;
	SECURITY_ATTRIBUTES SecurityAttr;
	bool ReturnResult = false;

	SecurityAttr.nLength = sizeof(SECURITY_ATTRIBUTES);
	SecurityAttr.bInheritHandle = TRUE;
	SecurityAttr.lpSecurityDescriptor = NULL;
	try {
		if (!CreatePipe(&hPipeRead, &hPipeWrite, &SecurityAttr, 0)) throw E_CREATE_PIPE;
		if (!SetHandleInformation(hPipeRead, HANDLE_FLAG_INHERIT, 0)) throw E_CREATE_PIPE;
		ZeroMemory(&StartInf, sizeof(StartInf));
		StartInf.cb = sizeof(StartInf);
		StartInf.dwFlags |= STARTF_USESTDHANDLES;
		StartInf.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
		StartInf.hStdError = GetStdHandle(STD_ERROR_HANDLE);
		StartInf.hStdOutput = hPipeWrite;
		ZeroMemory(&ProcessInf, sizeof(ProcessInf));
		// ��������� ������� ����������
		/******************************************************************************************************************************/
		if (CreateProcess(NULL, CompileCommand, NULL, NULL, TRUE, 0, NULL, CurDir, &StartInf, &ProcessInf)) {
			int counter = 0;
			int WaitRes = -1;
			DWORD Avail;
			//std::cout << std::endl << "Start compilation" << std::endl;
			//WaitRes = WaitForSingleObject(ProcessInf.hProcess, 3000000);
			while ((WaitRes = WaitForSingleObject(ProcessInf.hProcess, 100)) == WAIT_TIMEOUT && counter < 300) {
				counter++;
				if (PeekNamedPipe(hPipeRead, NULL, 0, NULL, &Avail, NULL) && Avail) {
					if (!ReadFile(hPipeRead, CompilerMsg, sizeof(CompilerMsg), &Read, NULL))
						LogMsg << "Reading pipe error: " << GetLastError() << endl;
					else CompilerMsg[Read] = '\0';
					LogMsg << CompilerMsg << endl;
				}
			}
			if (PeekNamedPipe(hPipeRead, NULL, 0, NULL, &Avail, NULL) && Avail) {
				if (Read != 0 && !ReadFile(hPipeRead, CompilerMsg, sizeof(CompilerMsg), &Read, NULL))
					LogMsg << "Reading pipe error: " << GetLastError() << endl;
				else CompilerMsg[Read] = '\0';
				LogMsg << CompilerMsg << endl;
			}
			if (WaitRes == WAIT_TIMEOUT && !TerminateProcess(ProcessInf.hProcess, 0)) {
				if (PeekNamedPipe(hPipeRead, NULL, 0, NULL, &Avail, NULL) && Avail) {
					if (!ReadFile(hPipeRead, CompilerMsg, sizeof(CompilerMsg), &Read, NULL))
						LogMsg << "Reading pipe error: " << GetLastError() << endl;
					else CompilerMsg[Read] = '\0';
					LogMsg << CompilerMsg << endl;
				}
				LogMsg << "Terminating process error: " << GetLastError() << endl;
			}

			//cout << endl << "Compilation finished" << endl;
			ReturnResult = true;
			throw E_SUCCESSFULL;
		}
		else throw E_CREATE_COMPILATION_PROCESS;
		/******************************************************************************************************************************/
	}
	catch (int Exception) {
		if (hPipeRead == NULL) CloseHandle(hPipeRead);
		if (hPipeWrite == NULL) CloseHandle(hPipeWrite);
		if (ProcessInf.hThread != NULL) CloseHandle(ProcessInf.hThread);
		if (ProcessInf.hProcess != NULL) CloseHandle(ProcessInf.hProcess);
		if (strcmp(Messages[Exception - 1], "") != 0)
			debugout << Messages[Exception - 1] << ": " << GetLastError() << endl;

		return ReturnResult;
	}
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// ������� ���������� ���������� �� ����� ���������������� ����������
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool PerformCompilation() {

	WIN32_FIND_DATA findFileData;
	//ostringstream filedir;
	//filedir << CurDir << "\sou"
	HANDLE hl = FindFirstFile("solutions\\*.cpp", &findFileData);
	if (hl == INVALID_HANDLE_VALUE) return false;
	do
	{
		////////////////////////////////////////////// ���������� ������ ���������� ///////////////////////////////////////////
		SYSTEMTIME st;
		GetLocalTime(&st);
		debugout << "File: " << findFileData.cFileName << endl;
		ostringstream os;
		os << "\"" << MVSDIR << "vcvarsall.bat\" & " << "cl.exe "
			<< "/I \"" << DIRECTXPATH << "include\" "
			<< "/I \"" << CurDir << "\\" << INCLUDEDIR << "\" "
			<< "/D_USRDLL /D_WINDLL \"" << CurDir << "\\" << SULUTIONDIR << "\\" << findFileData.cFileName << "\" \"" << CurDir << "\\include\\MyPlayerExport.cpp\" /link /DLL " 
			<< "/OUT:\""<< CurDir << "\\" << OUTDIR
			<< "\\" << setfill('0') << setw(2) << st.wDay << setw(2) << st.wMonth << st.wYear << "_" << setw(2) << st.wHour << setw(2) << st.wMinute << setw(2) << st.wSecond << "_" << st.wMilliseconds << ".dll\""
			<< " /MACHINE:X86 /SUBSYSTEM:WINDOWS /LIBPATH:\"" << DIRECTXPATH << "Lib\\x86\" "
			<< "/libpath:\"" << CurDir << "\\" << INCLUDEDIR << "\" /TLBID:1 /MD /OPENMP"
			//delete lib && obj files
			<< " & del " << OUTDIR << "\\*.lib"
			<<  "& del " << OUTDIR << "\\*.exp & del *.obj";
		string commandstr = os.str();
		os.str("");
		debugout << wstring(commandstr.begin(), commandstr.end()) << endl;
		if (!CompileSolution((char*)commandstr.c_str())) {
			debugout << "Internal tester error" << endl;
			return false;
		}
	} while (FindNextFile(hl, &findFileData) != 0);
	FindClose(hl);
	return true;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////