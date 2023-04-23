import { Component } from 'react';
import {StyleSheet, Text} from 'react-native'
import PropTypes from 'prop-types'

const styles = StyleSheet.create({
    text: {fontSize: 50}
})

// This isn't the cleanest API
class CountDown extends Component {
    static propTypes = {
        startTime: PropTypes.number.isRequired, // initial time on countDown clock
        onFinish: PropTypes.func.isRequired,    // function triggered when the count hits 0
        isRunning: PropTypes.bool.isRequired    // is it counting?
    }

    constructor(props) {
        super()
        this.state = {
            remainingSeconds: props.startTime * 60,
        }
    }

    resetState = () => {
        this.setState({remainingSeconds: this.props.startTime * 60})
    }

    decrease = () => {
        if (this.props.isRunning) {
            this.setState(prevState => ({remainingSeconds: prevState.remainingSeconds - 1}))
        }
    }

    componentDidMount() {
        // intentionally making it go 10x faster
        this.interval = setInterval(this.decrease, 100)
    }

    // since componentWillReceiveProps is deprecated, we can use componentDidUpdate
    // to check if the props have changed so we can update the state
    componentDidUpdate(prevProps, prevState) {
        if (prevState.remainingSeconds == 1) {
            this.props.onFinish()
        }

        if (prevProps.startTime != this.props.startTime) {
            this.setState({remainingSeconds: this.props.startTime * 60})
        }
    }

    componentWillUnmount() {
        clearInterval(this.interval)
    }

    render() {
        const displaySeconds = this.state.remainingSeconds % 60
        const displayMinutes = Math.floor(this.state.remainingSeconds/60)
        return (
            <Text style={styles.text}>
                {displayMinutes}:{displaySeconds.toString().padStart(2, '0')}
            </Text>
        )
    }
}

export { CountDown }
