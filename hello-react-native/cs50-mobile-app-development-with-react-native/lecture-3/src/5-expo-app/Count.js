import React from 'react'
import {StyleSheet, Text} from 'react-native'
// PropTypes:
// - validate types at runtime (in dev mode)
// - documents your API
import PropTypes from 'prop-types'

const styles = StyleSheet.create({
  text: {fontSize: 72},
})


class Count extends React.Component {
  static propTypes = {
    count: PropTypes.number.isRequired,
  }

  render() {
    return (
      <Text style={styles.text}>
        {this.props.count}
      </Text>
    )
  }
}

// you can also define PropTypes like this in class Components
// (I think for SFC you can only it this way)
// Count.propTypes = {
//   count: PropTypes.number.isRequired,
// }

export default Count
