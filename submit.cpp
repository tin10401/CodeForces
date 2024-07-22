#include <bits/stdc++.h>
#define all(x) (x).begin(), (x).end()
#define allr(x) (x).rbegin(), (x).rend()
#define ll long long
using namespace std;
const static int mod = 1e9 + 7;
const static string no = "NO\n";
const static string yes = "YES\n";
vector<vector<int>> dirs = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};

void solve()
{
    string s;
    cin >> s;
    string target = "hello";
    string ans = no;
    int i = 0, j = 0;
    while(i < s.size() && j < target.size())
    {
        if(s[i] == target[j]) j++;
        i++;
    }
    if(j == target.size()) ans = yes;
    cout << ans;
}


int main()
{
    solve();
}
