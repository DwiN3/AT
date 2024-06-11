// Generated from c:/Users/dwini/Desktop/Test2/AntlrCSharp/Math.g4 by ANTLR 4.13.1
import org.antlr.v4.runtime.Lexer;
import org.antlr.v4.runtime.CharStream;
import org.antlr.v4.runtime.Token;
import org.antlr.v4.runtime.TokenStream;
import org.antlr.v4.runtime.*;
import org.antlr.v4.runtime.atn.*;
import org.antlr.v4.runtime.dfa.DFA;
import org.antlr.v4.runtime.misc.*;

@SuppressWarnings({"all", "warnings", "unchecked", "unused", "cast", "CheckReturnValue", "this-escape"})
public class SpeakLexer extends Lexer {
	static { RuntimeMetaData.checkVersion("4.13.1", RuntimeMetaData.VERSION); }

	protected static final DFA[] _decisionToDFA;
	protected static final PredictionContextCache _sharedContextCache =
		new PredictionContextCache();
	public static final int
		SAYS=1, WORD=2, TEXT=3, WHITESPACE=4, NEWLINE=5;
	public static String[] channelNames = {
		"DEFAULT_TOKEN_CHANNEL", "HIDDEN"
	};

	public static String[] modeNames = {
		"DEFAULT_MODE"
	};

	private static String[] makeRuleNames() {
		return new String[] {
			"A", "S", "Y", "LOWERCASE", "UPPERCASE", "SAYS", "WORD", "TEXT", "WHITESPACE", 
			"NEWLINE"
		};
	}
	public static final String[] ruleNames = makeRuleNames();

	private static String[] makeLiteralNames() {
		return new String[] {
		};
	}
	private static final String[] _LITERAL_NAMES = makeLiteralNames();
	private static String[] makeSymbolicNames() {
		return new String[] {
			null, "SAYS", "WORD", "TEXT", "WHITESPACE", "NEWLINE"
		};
	}
	private static final String[] _SYMBOLIC_NAMES = makeSymbolicNames();
	public static final Vocabulary VOCABULARY = new VocabularyImpl(_LITERAL_NAMES, _SYMBOLIC_NAMES);

	/**
	 * @deprecated Use {@link #VOCABULARY} instead.
	 */
	@Deprecated
	public static final String[] tokenNames;
	static {
		tokenNames = new String[_SYMBOLIC_NAMES.length];
		for (int i = 0; i < tokenNames.length; i++) {
			tokenNames[i] = VOCABULARY.getLiteralName(i);
			if (tokenNames[i] == null) {
				tokenNames[i] = VOCABULARY.getSymbolicName(i);
			}

			if (tokenNames[i] == null) {
				tokenNames[i] = "<INVALID>";
			}
		}
	}

	@Override
	@Deprecated
	public String[] getTokenNames() {
		return tokenNames;
	}

	@Override

	public Vocabulary getVocabulary() {
		return VOCABULARY;
	}


	public SpeakLexer(CharStream input) {
		super(input);
		_interp = new LexerATNSimulator(this,_ATN,_decisionToDFA,_sharedContextCache);
	}

	@Override
	public String getGrammarFileName() { return "Math.g4"; }

	@Override
	public String[] getRuleNames() { return ruleNames; }

	@Override
	public String getSerializedATN() { return _serializedATN; }

	@Override
	public String[] getChannelNames() { return channelNames; }

	@Override
	public String[] getModeNames() { return modeNames; }

	@Override
	public ATN getATN() { return _ATN; }

	public static final String _serializedATN =
		"\u0004\u0000\u0005C\u0006\uffff\uffff\u0002\u0000\u0007\u0000\u0002\u0001"+
		"\u0007\u0001\u0002\u0002\u0007\u0002\u0002\u0003\u0007\u0003\u0002\u0004"+
		"\u0007\u0004\u0002\u0005\u0007\u0005\u0002\u0006\u0007\u0006\u0002\u0007"+
		"\u0007\u0007\u0002\b\u0007\b\u0002\t\u0007\t\u0001\u0000\u0001\u0000\u0001"+
		"\u0001\u0001\u0001\u0001\u0002\u0001\u0002\u0001\u0003\u0001\u0003\u0001"+
		"\u0004\u0001\u0004\u0001\u0005\u0001\u0005\u0001\u0005\u0001\u0005\u0001"+
		"\u0005\u0001\u0006\u0001\u0006\u0004\u0006\'\b\u0006\u000b\u0006\f\u0006"+
		"(\u0001\u0007\u0001\u0007\u0005\u0007-\b\u0007\n\u0007\f\u00070\t\u0007"+
		"\u0001\u0007\u0001\u0007\u0001\b\u0004\b5\b\b\u000b\b\f\b6\u0001\b\u0001"+
		"\b\u0001\t\u0003\t<\b\t\u0001\t\u0001\t\u0004\t@\b\t\u000b\t\f\tA\u0001"+
		".\u0000\n\u0001\u0000\u0003\u0000\u0005\u0000\u0007\u0000\t\u0000\u000b"+
		"\u0001\r\u0002\u000f\u0003\u0011\u0004\u0013\u0005\u0001\u0000\u0006\u0002"+
		"\u0000AAaa\u0002\u0000SSss\u0002\u0000YYyy\u0001\u0000az\u0001\u0000A"+
		"Z\u0002\u0000\t\t  D\u0000\u000b\u0001\u0000\u0000\u0000\u0000\r\u0001"+
		"\u0000\u0000\u0000\u0000\u000f\u0001\u0000\u0000\u0000\u0000\u0011\u0001"+
		"\u0000\u0000\u0000\u0000\u0013\u0001\u0000\u0000\u0000\u0001\u0015\u0001"+
		"\u0000\u0000\u0000\u0003\u0017\u0001\u0000\u0000\u0000\u0005\u0019\u0001"+
		"\u0000\u0000\u0000\u0007\u001b\u0001\u0000\u0000\u0000\t\u001d\u0001\u0000"+
		"\u0000\u0000\u000b\u001f\u0001\u0000\u0000\u0000\r&\u0001\u0000\u0000"+
		"\u0000\u000f*\u0001\u0000\u0000\u0000\u00114\u0001\u0000\u0000\u0000\u0013"+
		"?\u0001\u0000\u0000\u0000\u0015\u0016\u0007\u0000\u0000\u0000\u0016\u0002"+
		"\u0001\u0000\u0000\u0000\u0017\u0018\u0007\u0001\u0000\u0000\u0018\u0004"+
		"\u0001\u0000\u0000\u0000\u0019\u001a\u0007\u0002\u0000\u0000\u001a\u0006"+
		"\u0001\u0000\u0000\u0000\u001b\u001c\u0007\u0003\u0000\u0000\u001c\b\u0001"+
		"\u0000\u0000\u0000\u001d\u001e\u0007\u0004\u0000\u0000\u001e\n\u0001\u0000"+
		"\u0000\u0000\u001f \u0003\u0003\u0001\u0000 !\u0003\u0001\u0000\u0000"+
		"!\"\u0003\u0005\u0002\u0000\"#\u0003\u0003\u0001\u0000#\f\u0001\u0000"+
		"\u0000\u0000$\'\u0003\u0007\u0003\u0000%\'\u0003\t\u0004\u0000&$\u0001"+
		"\u0000\u0000\u0000&%\u0001\u0000\u0000\u0000\'(\u0001\u0000\u0000\u0000"+
		"(&\u0001\u0000\u0000\u0000()\u0001\u0000\u0000\u0000)\u000e\u0001\u0000"+
		"\u0000\u0000*.\u0005\"\u0000\u0000+-\t\u0000\u0000\u0000,+\u0001\u0000"+
		"\u0000\u0000-0\u0001\u0000\u0000\u0000./\u0001\u0000\u0000\u0000.,\u0001"+
		"\u0000\u0000\u0000/1\u0001\u0000\u0000\u00000.\u0001\u0000\u0000\u0000"+
		"12\u0005\"\u0000\u00002\u0010\u0001\u0000\u0000\u000035\u0007\u0005\u0000"+
		"\u000043\u0001\u0000\u0000\u000056\u0001\u0000\u0000\u000064\u0001\u0000"+
		"\u0000\u000067\u0001\u0000\u0000\u000078\u0001\u0000\u0000\u000089\u0006"+
		"\b\u0000\u00009\u0012\u0001\u0000\u0000\u0000:<\u0005\r\u0000\u0000;:"+
		"\u0001\u0000\u0000\u0000;<\u0001\u0000\u0000\u0000<=\u0001\u0000\u0000"+
		"\u0000=@\u0005\n\u0000\u0000>@\u0005\r\u0000\u0000?;\u0001\u0000\u0000"+
		"\u0000?>\u0001\u0000\u0000\u0000@A\u0001\u0000\u0000\u0000A?\u0001\u0000"+
		"\u0000\u0000AB\u0001\u0000\u0000\u0000B\u0014\u0001\u0000\u0000\u0000"+
		"\b\u0000&(.6;?A\u0001\u0006\u0000\u0000";
	public static final ATN _ATN =
		new ATNDeserializer().deserialize(_serializedATN.toCharArray());
	static {
		_decisionToDFA = new DFA[_ATN.getNumberOfDecisions()];
		for (int i = 0; i < _ATN.getNumberOfDecisions(); i++) {
			_decisionToDFA[i] = new DFA(_ATN.getDecisionState(i), i);
		}
	}
}