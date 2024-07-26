//https://cses.fi/problemset/task/1077/
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
    int n, k;
    cin >> n >> k;
    vector<int> arr(n);
    for(auto& it : arr) cin >> it;
    multiset<int> low, high;
    ll lowSum = 0, highSum = 0;
    for(int i = 0; i < n; i++)
    {
        if(i >= k)
        {
            if(arr[i - k] <= *low.rbegin()) lowSum -= arr[i - k], low.erase(low.find(arr[i - k]));
            else highSum -= arr[i - k], high.erase(high.find(arr[i  -k]));
        }
        low.insert(arr[i]);
        lowSum += arr[i];
        auto e = prev(low.end());
        lowSum -= *e;
        high.insert(*e);
        highSum += *e;
        low.erase(e);
        while(low.size() < high.size())
        {
            auto b = high.begin();
            low.insert(*b);
            lowSum += *b;
            highSum -= *b;
            high.erase(b);
        }
        if(i >= k - 1)
        {
            ll median = *low.rbegin();
            ll leftSum = median * low.size(), rightSum = median * high.size();
            cout << abs(leftSum - lowSum) +  abs(rightSum - highSum) << endl;
        }
    }
}

int main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr); 
    solve();
    return 0;
}

