#include <iostream>
#include <vector>

template<class T>
void merge(std::vector<T>& vec, int begin, int mid, int end) {
  int size1 = mid-begin+1;
  int size2 = end-mid;
  std::vector<T> L(size1);
  std::vector<T> R(size2);
  
  for(int i=0; i<L.size(); i++) {
    L[i] = vec[begin+i];
  }
  for(int j=0; j<R.size(); j++) {
    R[j] = vec[mid+j+1];
  }
  
  int i=0,j=0,k;
  for(k=begin; k<=end && i<L.size() && j<R.size(); k++) {
    if(L[i] <= R[j]) {
      vec[k] = L[i];
      i++;
    }
    else {
      vec[k] = R[j];
      j++;
    }
  }
  for(i=i; i<L.size(); i++) {
    vec[k] = L[i];
    k++;
  }
        
  for(j=j; j<R.size(); j++) {
    vec[k] = R[j];
    k++;
  }
}

template<class T>
void merge_sort(std::vector<T>& vec, int begin, int end) {
  if(begin < end) {
    int mid = (begin + end) / 2;
    merge_sort(vec, begin, mid);
    merge_sort(vec, mid+1, end);
    merge(vec, begin, mid, end);
  }
}

int main() {
  int n = 20;
  std::vector<double> vec(n);
  for (int i=0; i<n; i++) {
    vec[i] = drand48();
    std::cout << vec[i] << " ";
  }
  std::cout << std::endl;

  merge_sort(vec, 0, n-1);

  for (int i=0; i<n; i++) {
    std::cout << vec[i]	<< " ";
  }
  std::cout << std::endl;
}
