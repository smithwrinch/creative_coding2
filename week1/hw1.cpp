#include <iostream>
using namespace std;

int fibonacci(int n){
  if(n < 2){
    return n;
  } else{
    return fibonacci(n-1) + fibonacci(n-2);
  }

}

int main() {
  int n;
  cout << "Enter number for fibonacci: ";
  cin >> n;
  cout << "\nThe nth fibonacci number is: ";
  cout << fibonacci(n);
  cout << "\n";
  cout << "The sequence is:\n";
  for(int i =1; i <= n; i++){
    cout << fibonacci(i);
    cout << "\n";
  }
  return 0;
}
