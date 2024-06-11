import { TextInput } from 'react-native';
import React from 'react';
import Ionicons from 'react-native-vector-icons/Ionicons';
import { styles } from '../Styles/loginFormInputStyles';
import Animated, { FadeInUp } from 'react-native-reanimated';
import { HelperText } from 'react-native-paper';

export default function FormInput(props) {
  const {iconName, iconSize, text, value, handleOnChange, secue = false, errorMessage} = props;
  return (
    <>
    <Animated.View entering={FadeInUp.duration(1000).springify().randomDelay()} style={styles.inputBox}>
      <Ionicons name={iconName} size={iconSize} color= {errorMessage? 'red' : '#fff'} />
      <TextInput secureTextEntry={secue} value={value} onChangeText={handleOnChange} placeholder={text} placeholderTextColor={errorMessage? 'red' : '#fff'} style={[styles.inputText, errorMessage? styles.markText : null]}></TextInput>

    </Animated.View>
    {errorMessage && <Animated.View entering={FadeInUp.duration(1000).springify().randomDelay()} style={styles.errorMessageBox}>
      <HelperText type="error" style={styles.errorMessage}>
          {errorMessage}
      </HelperText></Animated.View>}
      </>
  )
}