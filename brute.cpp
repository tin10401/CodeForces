#include <bits/stdc++.h>
using namespace std;

int main () {
  ios_base::sync_with_stdio(0); cin.tie(0);
  int T;
  cin >> T;
  while (T--) {
    int n;
    cin >> n;
    vector<int> c(n);
    for (int i = 0; i < n; i++) {
      vector<int> a(n);
      for (int& x: a) cin >> x;
      while (!a.empty() && a.back() == 1) c[i]++, a.pop_back();
    }
    sort(c.begin(), c.end());

    int ptr = 0;
    int ans = 0;
    while (ptr < n) {
      while (ptr < n && c[ptr] < ans) ptr++;
      if (ptr < n) ans++, ptr++;
    }
    cout << ans << '\n';
  }
}

