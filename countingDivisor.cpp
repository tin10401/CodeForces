#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
using namespace __gnu_pbds;
using namespace std;
typedef tree<pair<int, int>, null_type, less<pair<int, int>>, rb_tree_tag, tree_order_statistics_node_update> ordered_set;
#define all(x) begin(x), end(x)
#define allr(x) rbegin(x), rend(x)
#define ll unsigned long long
#define int long long
#define pb push_back
#define vi vector<int>
#define pii pair<int, int>
#define vpii vector<pair<int, int>>
#define f first
#define s second
#define ar(x) array<int, x>
const static int INF = 1e18;
const static int mod = 1e9 + 7;
const static string no = "NO\n";
const static string yes = "YES\n";
const vector<vector<int>> dirs = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};

int divisor[1000005] = {};


int modExpo(int base, int exp)
{
    int res = 1;
    while(exp)
    {
        if(exp & 1) res = (res * base) % mod;
        base = (base * base) % mod;
        exp >>= 1;
    }
    return res;
}

void calculateDivisor()
{
    for(int i = 1; i < 1000005; i++)
    {
        for(int j = i; j < 1000005; j += i) divisor[j]++;
    }
}

void solve()
{
    int n;
    cin >> n;
    calculateDivisor();
    for(int i = 0; i < n; i++)
    {
        int num;
        cin >> num;
        cout << divisor[num] << endl;
    }

}

signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr); 
    solve();
    return 0;
}

