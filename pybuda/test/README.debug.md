
*Test specific environment variables that can be used to fine tune default behavior of PyBuda tests.*

## Parameters
 * RANDOM\_TEST\_COUNT: Number of random tests to be generated and executed. The parameter generate test_index in range from 0 to RANDOM\_TEST\_COUNT-1. (default: 5)
 * RANDOM\_TESTS\_SELECTED: Limiting random tests to only selected subset defined as comma separated list of test indexes. E.x. "3,4,6". Default is no limitation if not specified or empty.
