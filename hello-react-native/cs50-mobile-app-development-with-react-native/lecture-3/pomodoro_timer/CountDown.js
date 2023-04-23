import { Component } from 'react';
import {StyleSheet, Text} from 'react-native'
import PropTypes from 'prop-types'

const styles = StyleSheet.create({
    text: {fontSize: 50}
})

class CountDown extends Component {
    static propTypes = {
        startTime: PropTypes.number.isRequired,
        message: PropTypes.string.isRequired,
        onFinish: PropTypes.func.isRequired,
        isRunning: PropTypes.bool.isRequired
    }

    constructor(props) {
        super()
        this.state = {
            remainingSeconds: props.startTime * 60,
        }
    }

    // componentWillReceiveProps(nextProps) {
    //     this.setState({remainingSeconds: props.startTime * 60,})
    // }

    decrease = () => {
        if (this.props.isRunning) {
            this.setState(prevState => ({remainingSeconds: prevState.remainingSeconds - 1}))
        }
    }

    componentDidMount() {
        this.interval = setInterval(this.decrease, 100)
    }

    // shouldComponentUpdate(_, nextState) {
    //     return nextState.remainingSeconds >= 0
    // }

    componentDidUpdate(_, prevState) {
        if (prevState.remainingSeconds == 1) {
            this.props.onFinish()
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
                {this.props.message}: {displayMinutes}:{displaySeconds.toString().padStart(2, '0')}
            </Text>
        )
    }
}

export { CountDown }
