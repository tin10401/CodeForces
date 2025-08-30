// debug.h
#pragma once
#include <bits/stdc++.h>
using namespace std;

#ifdef DEBUG_AUTO_FLUSH
struct _AutoFlush { _AutoFlush(){ cerr.setf(ios::unitbuf); cout.setf(ios::unitbuf); } } _autoFlush;
#endif

#ifdef LOCAL

// --------- Printers (defined BEFORE debug_out) ---------

// __int128
inline ostream& operator<<(ostream& os, __int128 x){
    if (x == 0){ os << '0'; return os; }
    if (x < 0){ os << '-'; x = -x; }
    char s[50]; int n = 0;
    while (x){ s[n++] = char('0' + int(x % 10)); x /= 10; }
    while (n--) os << s[n];
    return os;
}

// pair
template<class A, class B>
inline ostream& operator<<(ostream& os, const pair<A,B>& p){
    return os << "{" << p.first << " , " << p.second << "}";
}

// pretty map
template<class K, class V>
inline ostream& operator<<(ostream& os, const map<K,V>& m){
    os << '{'; bool first = true;
    for (auto const& [k,v] : m){ os << (first ? "" : " , ") << k << " : " << v; first = false; }
    return os << '}';
}

// detect iterables
template<class C>
struct _dbg_iterable {
    template<class T>
    static auto test(int) -> decltype(begin(declval<T&>()), end(declval<T&>()), true_type{});
    template<class> static auto test(...) -> false_type;
    static constexpr bool value = decltype(test<C>(0))::value;
};

// treat std::string as scalar (not a container)
template<class T>
inline constexpr bool _dbg_is_string = is_same_v<decay_t<T>, string>;

// exclude C-strings and char arrays from generic container printer
template<class T>
inline constexpr bool _dbg_is_cstr =
    is_same_v<decay_t<T>, const char*> || is_same_v<decay_t<T>, char*>;

template<class T>
inline constexpr bool _dbg_is_char_array =
    is_array_v<remove_reference_t<T>> &&
    is_same_v<remove_cv_t<remove_extent_t<remove_reference_t<T>>>, char>;

// generic container printer (NOT for string/char*/char[N])
template<class C, enable_if_t<
    _dbg_iterable<C>::value &&
    !_dbg_is_string<C> &&
    !_dbg_is_cstr<C> &&
    !_dbg_is_char_array<C>, int> = 0>
inline ostream& operator<<(ostream& os, const C& cont){
    os << '{'; bool first = true;
    for (auto const& x : cont){ os << (first ? "" : " , ") << x; first = false; }
    return os << '}';
}

// --------- debug(...) macro ---------
#define debug(...) debug_out(#__VA_ARGS__, __VA_ARGS__)

inline void debug_out(const char*) { cerr << '\n'; }

template<class T, class... R>
inline void debug_out(const char* names, T&& v, R&&... r){
    const char* comma = strchr(names, ',');
    cerr << "[" << (comma ? string(names, comma) : string(names)) << " = " << v << "]";
    if constexpr (sizeof...(r)){
        cerr << ", ";
        debug_out(comma ? comma+1 : names, std::forward<R>(r)...);
    } else {
        cerr << '\n';
    }
}

// --------- timers & memory usage ---------

// Use in pairs: startClock; ...; endClock;
#define startClock do { clock_t _dbg_tStart = clock();
#define endClock   cout << fixed << setprecision(10) \
                    << "\nTime Taken: " \
                    << double(clock() - _dbg_tStart) / CLOCKS_PER_SEC \
                    << " seconds\n"; } while(0)

#if defined(__linux__)
  #include <sys/resource.h>
  #include <sys/time.h>
  inline void printMemoryUsage(){
      rusage u{}; getrusage(RUSAGE_SELF, &u);
      cerr << "Memory: " << (u.ru_maxrss / 1024.0) << " MB\n";
  }
#else
  inline void printMemoryUsage() {}
#endif

#else  // !LOCAL

#define debug(...)
#define startClock do {} while(0)
#define endClock   do {} while(0)
inline void printMemoryUsage() {}

#endif

