// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use lmm_agent::runtime::AutoAgent;

#[test]
fn default_settings() {
    let runner = AutoAgent::default();
    assert!(runner.execute);
    assert!(!runner.browse);
    assert_eq!(runner.max_tries, 1);
}

#[test]
fn build_fails_without_agents() {
    let result = AutoAgent::default().build();
    assert!(result.is_err());
}
