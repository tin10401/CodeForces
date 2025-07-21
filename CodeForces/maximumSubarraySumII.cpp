//https://cses.fi/problemset/task/1644/
#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
using namespace __gnu_pbds;
using namespace std;
typedef tree<pair<int, int>, null_type, less<pair<int, int>>, rb_tree_tag, tree_order_statistics_node_update> ordered_set;
#define all(x) (x).begin(), (x).end()
#define allr(x) (x).rbegin(), (x).rend()
#define ll long long
const static int mod = 1e9 + 7;
const static string no = "NO\n";
const static string yes = "YES\n";
const vector<vector<int>> dirs = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};

ll modExpo(ll base, ll exp)
{
    ll res = 1;
    while(exp)
    {
        if(exp & 1) res = (res * base) % mod;
        base = (base * base) % mod;
        exp >>= 1;
    }
    return res;
}


void solve()
{
    int n, a, b;
    cin >> n >> a >> b;
    vector<ll> prefix(n + 1);
    for(int i = 1; i <= n; i++)
    {
        int num;
        cin >> num;
        prefix[i] = prefix[i - 1] + num;
    }
    ll res = LLONG_MIN;
    multiset<ll> s;
    for(int i = a; i <= n; i++)
    {
        if(i > b) s.erase(s.find(prefix[i - b - 1]));
        s.insert(prefix[i - a]);
        res = max(res, prefix[i] - *s.begin());
    }
    cout << res << endl;
}

int main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr); 
    solve();
    return 0;
}

