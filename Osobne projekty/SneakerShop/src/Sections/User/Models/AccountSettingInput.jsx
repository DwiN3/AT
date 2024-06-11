import { Text, TextInput } from 'react-native';
import React from 'react';
import { styles } from '../Styles/ChangeInformationStyle';
import Animated, { FadeInUp } from 'react-native-reanimated';
import { HelperText } from 'react-native-paper';

export default function AccountSettingInput(props) {
  const { text, value, handleOnChange, secue = false, errorMessage } = props;

  return (
    <>
      <Animated.View entering={FadeInUp.duration(1000).springify().randomDelay()} style={styles.inputBackground}>
        <Text style={styles.textHint}>{text}</Text>
        <TextInput style={[styles.input, errorMessage? styles.inputError : null]} secureTextEntry={secue} value={value} onChangeText={handleOnChange} placeholderTextColor={errorMessage? 'red' : '#000'}></TextInput>
      </Animated.View>

        {errorMessage &&
          <Animated.View entering={FadeInUp.duration(1000).springify().randomDelay()} style={styles.errorMessageBox}>
            <HelperText type="error" style={styles.errorMessage}>
              {errorMessage}
            </HelperText>
          </Animated.View>
        }
    </>
  )
}