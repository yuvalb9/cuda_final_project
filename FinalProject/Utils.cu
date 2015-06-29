#include "Utils.cuh"


void printBoard(int boardHeight, int boardWidth, char* board)
{
	for (int i = 0; i<boardHeight; i++)
	{
		for (int j = 0; j<boardWidth; j++)
		{
			// Keep the numbers in the range of 0 to COLORS-1
			printf("%d ", board[i*boardWidth + j]);
		}
		printf("\n");
	}
}

void outputBoardToFile(char* board, int boardHeight, int boardWidth, int colors, const char* filePath)
{
	std::ofstream myfile(filePath);
	if (myfile.is_open())
	{
		myfile << "P6 ";
		myfile << boardWidth << " ";
		myfile << boardHeight << " ";	
		myfile << 255 << "\n";
		
		for (int i = 0; i < boardHeight*boardWidth; i++)
		{
			switch ((int)(board[i]))
			{
			case 0:
				myfile << char(255) << char(0) << char(0);
				break;
			case 1:
				myfile << char(255) << char(128) << char(0);
				break;
			case 2:	
				myfile << char(255) << char(255) << char(0);
				break;
			case 3:
				myfile << char(128) << char(255) << char(0);
				break;
			case 4:
				myfile << char(0) << char(255) << char(0);
				break;
			case 5:
				myfile << char(0)<< char(255) << char(128);
				break;
			case 6:
				myfile << char(0) << char(255) << char(255);
				break;
			case 7:
				myfile << char(0) << char(128) << char(255);
				break;
			case 8:
				myfile << char(0) << char(0) << char(255);
				break;
			case 9:
				myfile << char(127) << char(0) << char(255);
				break;
			case 10:
				myfile << char(255) << char(0) << char(255);
				break;
			case 11:
				myfile << char(255)<< char(0) << char(127);
				break;
			case 12:
				myfile << char(128) << char(128) << char(128);
				break;
			case 13:
				myfile << char(64) << char(128) << char(0);
				break;
			case 14:
				myfile << char(0) << char(64) << char(128);
				break;
			case 15:
				myfile << char(128) << char(0) << char(64);
				break;
			default:
				break;
			}
			
		}
		myfile.close();
	}
	else printf("Unable to open file");
}


unsigned int nextPow2(unsigned int x)
{
	--x;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	return ++x;
}